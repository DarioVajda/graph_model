# RelBench × GTLM — implementation plan

**Status:** in progress. Written 2026-07-26; revised 2026-07-27 twice — sampler switched
from hand-rolled to PyG `NeighborLoader` (§2.1, §5.2), and the headline moved from `rel-f1`
to `rel-trial` (§1.1).

**Scope.** Two datasets with two different jobs:

* **`rel-f1` — the debugging target.** 0.7 MB, 9 tables, three entity tasks. Everything
  through the first graph-vs-flat number is developed here, because a temporal-leakage bug
  is far easier to find in 9 tables than in 15 and `driver-dnf` has the cheapest eval path
  in the suite. It does **not** carry the scientific claim (§1.1).
* **`rel-trial` — the headline target.** 548 MB, 15 tables, 5.4M rows;
  `study-outcome` / `study-adverse` / `site-success`. This is where the GTLM thesis is
  actually testable.

One model per task. The *method* is task- and dataset-agnostic (§5.0): a new
`(dataset, task)` pair runs with **zero code changes** — no hand-written column spec, prompt
or budget anywhere. That constraint is what makes the two-target split free rather than
double work.

---

## 1. Why this experiment, and what would count as a result

RelBench is *relational deep learning*: predict a label for an entity **at a timestamp**,
from the rows of a normalized multi-table database reachable through primary/foreign keys.
The standard pipeline (RDL) encodes every row into a fixed-width vector with
`pytorch_frame` (categorical embeddings + a frozen sentence encoder for text columns) and
runs a heterogeneous GNN over temporally-sampled subgraphs.

That is *exactly* the compressive bottleneck GTLM exists to remove — one row, however many
text columns it has, becomes one vector before any cross-row attention happens. So the
thesis transfers cleanly: **feed the sampled rows as full text, put the fkey topology into
the attention bias, keep one model.**

Anchors, read off the source PDFs (Wydmuch, Borchmann & Graliński, *Tackling prediction
tasks in relational databases with LLMs*, arXiv:2411.11829v1, Tables 1/2/5/6):

| rel-f1 task | LightGBM | RDL (GNN) | Llama-3.2-1B | 1B + MLP head | Llama-3.2-3B | 3B + MLP |
|---|---:|---:|---:|---:|---:|---:|
| `driver-dnf` (AUROC ↑) | 68.85 | 72.62 | 65.81 | 78.41 | 80.03 | 82.33 |
| `driver-top3` (AUROC ↑) | 73.93 | 75.54 | **88.47** | 87.36 | 87.11 | 89.70 |
| `driver-position` (MAE ↓) | 4.170 | 4.022 | — | 3.539 | — | 3.092 |

**Read these carefully — four caveats change what "beating them" means.**

1. **Nothing is fine-tuned.** Their LLM columns are a *frozen base model* read out either by
   the probability of the `"1"` token ("metric-aware inference") or by a 2-layer MLP
   (hidden width 10) trained on one token embedding from ≤1e5 documents. We fine-tune with
   LoRA. So their number is an **external anchor, not a matched control** — our matched
   control is our own flat serialization (§8), and that distinction has to be stated in
   any writeup.
2. **The 80.03 I quoted earlier is the 3B model, not the 1B.** The 1B without a head is
   **65.81** on `driver-dnf` — *below* LightGBM. The plan's earlier framing ("the 1B LLM
   baseline already beats RDL") was wrong: with a frozen 1B and no head it loses to a GBT.
3. **rel-f1 is the paper's own admitted contamination case.** They write that on rel-f1
   the models are "mostly relying on their pre-existing factual knowledge of Formula-1,"
   and that "adding more information seems to only confuse models." The evidence is in
   their per-config table: the best 1B `driver-top3` result (88.47) comes from
   `n_inc=8, n_rel=0, d=0, n_nest=0` — **eight in-context examples and the driver's own row,
   with zero rows retrieved from the database.** Their best `driver-dnf` 1B config (65.81)
   is `n_inc=0, n_rel=16, d=0, n_nest=0` — the driver's own 16 most recent *labels*, again
   no database rows. Database traversal (`d=1`) makes both *worse*.

So `driver-top3` at 88.47 is not a relational-reasoning result; it is Llama knowing that
Hamilton and Vettel qualify near the front. Any comparison that does not control for this
measures pretraining memorization. §8.6 makes that control a first-class arm — and it is
arguably the most publishable single thing in this experiment, because it is a criticism
their own paper invites but does not run.

4. **Every cell above is a per-task best-of-13, selected on validation.** Their document
   parameters `(n_inc, n_rel, d, n_nest)` are re-tuned for each task independently. We
   report against **both** their per-task envelope and their best *single* configuration,
   because we produce both numbers ourselves (§5.0 B). Recomputed from their per-config
   table (Table 5, 1B, the 10 tasks with a complete grid):

   | | mean AUROC over 10 tasks | `driver-dnf` | `driver-top3` |
   |---|---:|---:|---:|
   | per-task best-of-13 (their headline) | 68.34 | 65.81 | 88.47 |
   | best single config, `(n_inc=8, n_rel=8, d=0, n_nest=0)` | 62.66 | 59.24 | 77.40 |

   **Per-task tuning is worth +5.7 AUROC on average to them** — larger than the entire
   RDL-vs-LightGBM gap, and a useful measure of how much of any published RelBench number is
   construction tuning rather than method. Match rows to rows: our `per_task` number goes
   against 65.81 / 88.47, our `global` number against 59.24 / 77.40.

   Note also *which* config wins on average: `d=0, n_nest=0` — **no database traversal at
   all**, tied with its `d=1` sibling. Averaged over tasks, their retrieved rows do not pay
   for themselves. That is the gap our construction (§5.4) is trying to open.

**Win condition, in kgqa's matched-input style (do not chase the leaderboard):**

0. **Read §1.1 first: the headline runs on `rel-trial`, not `rel-f1`.** The conditions
   below are stated per dataset and apply to whichever is being reported.
1. **GTLM ≥ our own flat-text control** at byte-identical supervision — the same sampled
   rows, the same base model, the same schedule, no structural biases. This is the result
   that transfers to the rest of the repo, and it is the one kgqa has *not* won on
   Freebase (74.1 vs 74.9 F1). RelBench is a genuinely different test of the same claim:
   here the neighborhood is a *set of records with numeric/categorical fields and a
   temporal ordering*, which is where an explicit topology channel should pay more than it
   does on near-shortest-path KG subgraphs.
2. **GTLM ≥ RDL** at comparable input budget — the pipeline-vs-single-model comparison.
3. **GTLM ≥ the LLM-baseline paper at matched inputs** — i.e. beat it in the
   `sampling: paper_match` condition (§5.4), where our document contains the same rows
   theirs does. Beating their headline with a *richer* neighborhood is a weaker claim and
   should be reported separately from the matched one.
4. **The contamination control (§8.6) is reported alongside every rel-f1 number.** Without
   it, `driver-top3` is uninterpretable.

Only after 1–3: bigger subsets, text-heavy databases, and (maybe) recommendation tasks.

**Explicit non-goals for phase 1:** link-prediction / recommendation tasks (`MAP@k` needs a
ranking head over thousands of candidates — a different eval architecture), the
`AutoCompleteTask` family, and any leaderboard submission.

### 1.1 Why the headline moved to `rel-trial` (decided 2026-07-27)

**What rel-f1 cannot test.** GTLM's thesis is that collapsing a row's *text* into one vector
before any cross-row attention destroys information. Look at what a rel-f1 row holds:
`results` is `grid, position, positionOrder, points, laps, milliseconds, fastestLap, rank,
statusId` — entirely numeric. The only free text in the database is names on the dimension
tables (`drivers.forename/surname`, `circuits.name`, `constructors.name`), which is exactly
the memorizable content §8.6 exists to *control*. `pytorch_frame` encodes numerics well, so
on rel-f1 the compressive bottleneck barely bites. rel-f1 can settle win condition 1 (does
the topology channel beat flat serialization on identical evidence) and prove the machinery
works. It cannot settle the thesis.

**It is also the suite's contamination case.** Same six columns as §1's anchors table:

| task | LightGBM | RDL | **Llama-1B** | 1B+MLP | Llama-3B | 3B+MLP |
|---|---:|---:|---:|---:|---:|---:|
| rel-f1 `driver-top3` | 73.93 | 75.54 | **88.47** | 87.36 | 87.11 | 89.70 |
| rel-trial `study-outcome` | 70.09 | 68.60 | **55.72** | 68.38 | 59.17 | 70.82 |

A frozen 1B scores 88.47 on `driver-top3` and 55.72 on `study-outcome`. The first is Llama
knowing Formula 1; the second is what an uncontaminated benchmark looks like — barely above
chance. Any rel-f1 headline has to argue its way out of that gap; a rel-trial headline does
not.

**rel-trial is where the thesis is testable, and there is headroom.** 15 tables, 5.4M rows,
140 columns. `studies` carries `brief_title`, `official_title`, `detailed_descriptions`,
`brief_summaries`, `baseline_population`, `limitations_and_caveats`, `biospec_description`;
`eligibilities` carries criteria text; `outcomes` carries `title`/`description`/`population`.
These are paragraphs, and RDL must push each through a frozen sentence encoder into one
vector. Meanwhile **every LLM approach loses to LightGBM there** (best is 3B+MLP at 70.82 vs
70.09; RDL at 68.60 is *below* the GBT), so a fine-tuned GTLM clearing 70 is a real result
rather than a memorization artifact.

**Cost is mostly download, not training.** `study-outcome` is **11,994 / 960 / 825** —
essentially `driver-dnf`'s 11,411 / 566 / 702. The database is 548 MB rather than 0.7 MB,
and `study-adverse` is larger at 43,335 / 3,596 / 3,098.

**Two risks to measure on rel-trial specifically, not assume:**

1. **Token budget inverts the node budget.** The baseline's rel-trial documents run
   **1,571–42,720 tokens** (their Table 7) against rel-f1's 211–4,314. Text-heavy rows mean
   fewer nodes fit in context, so "text per row" and "nodes per graph" trade off directly.
   If the graph collapses to ~16 nodes the topology story weakens, and `max_node_chars`
   stops being cosmetic. Measure at the §6 document-dump gate on trial, before committing.
2. **Scale arrives early.** 5.4M rows makes §10's index-memory and prep-wall-clock concerns
   present-tense rather than future work.

**What this changes downstream.** §9's configs `004`–`007` were scoped to rel-f1; the
construction ablation's GPU budget moves to rel-trial, and rel-f1 keeps only the cheap
canaries and the first graph-vs-flat pair. §10's "phase 2" is no longer later work — it is
the headline. Nothing in §2–§7 changes: the pipeline is dataset-agnostic by construction
(§5.0 A), which is what makes a two-target split free rather than double work.

### 1.2 Baseline-compatibility audit (2026-07-27, before any GPU commit)

Everything built through M5 was checked against the relbench 2.1.2 source and against the
baselines' own runner (`snap-stanford/relbench:examples/gnn_entity.py`,
`examples/lightgbm_entity.py`). The question was narrow: **would a number produced by this
pipeline be comparable to the anchors in §1?**

**Verified identical to the baselines.**

| # | Property | Evidence |
|---|---|---|
| 1 | **Input data.** We build the graph from `get_dataset(...).get_db()`, i.e. `upto_test_timestamp=True` | `gnn_entity.py` uses the same default. Rows after `test_timestamp` are input for nobody |
| 2 | **Task tables.** Official downloaded parquets, not locally recomputed | Row counts `driver-dnf` 11,411/566/702 and `study-outcome` 11,994/960/825 match Table 4 of arXiv:2411.11829 **exactly** — so relbench 2.1.2 serves the same tables the anchors were computed on |
| 3 | **Evaluation index.** Val/test in table order, never shuffled | `gnn_entity.py` passes `shuffle=(split=="train")`; `EntityTask.evaluate` compares positionally and checks only *length* |
| 4 | **Metric + selection.** `roc_auc`, higher-is-better, from `relbench.metrics` | `driver-dnf` and `study-outcome` both declare `metrics = [average_precision, accuracy, f1, roc_auc]`; the baselines tune on `roc_auc` |
| 5 | **Best-checkpoint restore.** `GraphTrainerV2._load_best_model` | The baselines `model.load_state_dict(state_dict)` before final eval. M6 must use `GraphTrainerV2`, which is the trainer with the two-part reload fix |
| 6 | **No answer leak.** The prompt node is strictly causal | `flex_kernel.py:223-227` and `structural_mask.py:56-61` both gate bidirectionality on *both* endpoints being non-prompt nodes, so the answer token is invisible to the position that predicts it |
| 7 | **No differential leakage.** We can see no column RDL cannot | `datasets/trial.py:78` — relbench already drops `overall_status`/`why_stopped`/`completion_date` from `studies`; our renderer reads `db.table_dict[...].df.columns`, i.e. exactly what survives |
| 8 | **Genuine estimation, not retrieval.** The label rows are outside the filter | `StudyOutcomeTask`'s window is `oa.date > t AND oa.date <= t + 365d`; the sampler admits `time <= t`. Same for `driver-dnf` (`re.date > t`) |
| 9 | **Anchors themselves.** All six numbers in §1.1's table | Re-read verbatim off the source PDF, lines 385-387: `70.09 68.60 55.72 68.38 59.17 70.82` / `68.85 72.62 65.81 78.41 80.03 82.33` / `73.93 75.54 88.47 87.36 87.11 89.70` |

**Two bugs found and fixed.**

* **The sample caps were not in the cache key.** `max_train_samples` / `max_val_samples`
  stride the build (`data.py`), so they change the built bytes — but `_CACHE_KEYS` omitted
  them. This had already happened: both M5 caches held **201 of 11,411** train rows and
  **114 of 566** val rows under a key a full build would also produce, so a full-scale
  `data_prep` would have found `_is_built()` true, skipped, and trained on 1.8% of the data
  while reporting it as a real number. The ordering assert could not catch it — strided
  row_ids are still monotonic. Both keys are now in `_CACHE_KEYS`, `assert_row_order` grew a
  `contiguous=` check that requires test (and uncapped val) to be exactly `0..n-1`, and the
  two poisoned caches were deleted.
* **`val_subsample` / `test_subsample` were dead flags.** Never read by anything. A dead
  `test_subsample` is worse than a missing one — it reads, in a sweep config, as though the
  test split *was* subsampled. `validate()` now rejects both, and the rejection message says
  why test must never be subsampled: `task.evaluate` compares to the full table positionally.

**Three asymmetries that are not bugs but must be stated in any writeup.**

1. **We feed far less than RDL, not "comparable input budget".** RDL's defaults are
   `num_neighbors=128`, `num_layers=2` → `[128, 64]`, i.e. up to ~8,300 rows per seed. We
   feed `max_nodes` = 24–64 *content* rows. §1's win condition 2 said "at comparable input
   budget"; that is not what we are doing and the phrasing is now wrong. The honest claim is
   **"GTLM ≥ RDL at a 100× smaller neighborhood"** — which is a stronger result if we win
   and an expected one if we lose. Do not present it as a matched comparison.
2. **RDL's default `temporal_strategy` is `"uniform"`; our default is `"recent"`** (PyG
   `"last"`). Both are available on both sides. State which we report.
3. **The flat control is causal-only.** In the flat arm every token belongs to the prompt
   node, so `allowed = causal` — it gets no bidirectional attention, while the graph arm
   encodes its content nodes bidirectionally. So "graph > flat" conflates the *bias channel*
   with *bidirectional encoding of identical content*. This is the repo-wide convention
   (kgqa, tag_benchmarks), not a new problem, but on RelBench it is worth splitting: the
   clean isolation of the bias channel is a **third arm — multi-node, `--no-spd
   --no-magnetic`** — which is expressible today (`arm_name=graph`, both biases off, labelled
   `no-spd+magnetic`) and costs one extra run per pair. **Recommended for the M6 pair.**

**Also noted.** LightGBM's baseline input is only the entity table's own row
(`lightgbm_entity.py` does a single `merge`, no neighbor aggregation) — so our `max_nodes=1`
flat arm is approximately LightGBM's input, which makes 68.85 / 70.09 a meaningful floor for
that ablation point rather than an unrelated number.

**One protocol consequence worth remembering.** `driver-dnf` has
`num_eval_timestamps = 40`, so test seeds run from 2010-03 to 2013-03 while the input DB
stops at `test_timestamp` (2010-01-01). Late test seeds are predicted from history up to
three years stale. That is the benchmark's design and RDL suffers it identically — but it
caps how much a *recency*-biased sampler can help on rel-f1, and is one more reason the
headline lives on rel-trial (`study-outcome` has `num_eval_timestamps = 1`).

---

## 2. What is genuinely new here vs. what is reuse

Reuse, unchanged:

* `TextGraphDataset` (nx graph in → tokenized + SPD/RRWP/magnetic features + `.gtds`
  chunks out), `GraphCollatorV2`, `GraphTrainerV2`, `GTLMLlamaForCausalLM`, `v2-flex`.
* The `sweep` runner contract (`RunConfig` → CLI flags → one JSONL line per run).
* The tag_benchmarks data-prep shape: `--mode data_prep` builds and caches every split,
  `--mode train` refuses to build and fails loudly if the cache is missing.
* The kgqa `question_node: isolated` layout, and its train-on-train /
  select-on-val / score-test-once protocol.

Genuinely new, in order of risk:

1. **A relational→`HeteroData` build and the sampling wiring** (§5). The *sampler itself* is
   PyG's — see §2.1, which corrects an earlier version of this plan that proposed writing
   one from scratch. What is still ours: converting a relbench `Database` into a
   `HeteroData` without `torch_frame`, mapping sampler output back to `(table, row, hop)`,
   and the three selection policies PyG cannot express.
2. **Row → text** (§6). New surface, and the single biggest lever on final numbers
   (cf. kgqa data-format v3 being worth +4.6 F1).
3. **A score-based eval path** (§7). Every existing experiment scores exact match over
   generated/teacher-forced tokens. RelBench needs AUROC/AP (a *ranking* over a continuous
   score) and MAE/R² (a real number). This is new code in the trainer, not a metric rename.

### 2.1 Correction: use PyG's sampler, do not write one

An earlier revision of this plan proposed a ~200-line hand-rolled temporal sampler, on two
grounds. One was weak and one was false. Recording both, because the reasoning matters more
than the conclusion:

* *"`pyg-lib` is absent and `torch-sparse` is broken, so `NeighborLoader` has no backend."*
  True of the venv as it stands — verified 2026-07-27:

  ```
  torch 2.11.0+cu130   pyg 2.7.0
  WITH_PYG_LIB False   WITH_TORCH_SPARSE False   WITH_SAMPLED_OP False
  ```

  `torch-sparse` and `torch-scatter` are compiled against CUDA 12.8 and PyG disables them at
  import; `NeighborSampler` then hard-raises `"requires either 'pyg-lib' or 'torch-sparse'"`
  (`neighbor_sampler.py:512`). But a matching wheel is published —
  `pyg_lib-0.8.0+pt211cu130-cp310-abi3-manylinux_2_28_x86_64.whl` — so this is one
  `pip install`, not a reason to reimplement. See §3.
* *"PyG's uniform sampler cannot express the most-recent-k policy."* **Wrong.**
  `NeighborLoader(..., temporal_strategy='last')` samples exactly "the last `num_neighbors`
  that fulfil temporal constraints" (`loader/neighbor_loader.py:155-161`). Together with
  `time_attr` + per-seed `input_time` and per-relation `num_neighbors` on `HeteroData`,
  that *is* §5.4's `recent` policy, plus `uniform`, plus the fanout design.

Owning less of this code is a real win, not just less typing: §5.3 test 1 (no sampled row
with `time > ts`) is the test whose silent failure voids the entire experiment, and PyG's
temporal filter is battle-tested C++ that RDL itself runs on. Keeping our sampling
comparable to RDL's also makes the RDL comparison (win condition 2) more defensible.

**Independently worth fixing:** the broken `torch-scatter`/`torch-sparse` silently degrade
any PyG code elsewhere in this repo. Reinstalling them at `+pt211cu130` is not needed for
this experiment (`pyg-lib` alone satisfies the sampler) but should be raised separately.

---

## 3. Phase 0 — environment

All of it runs on the **login node** — the wheel index and the dataset download both need
outbound internet, which compute nodes do not have.

### 3.1 `relbench` itself

```bash
# relbench's own code is pure pandas/duckdb/pooch — torch_frame is only needed by
# relbench.modeling, which we bypass entirely.
.venv/bin/pip install --no-deps relbench==2.1.2
.venv/bin/pip install pooch duckdb          # the only two deps we don't already have
```

**Install with `--no-deps` deliberately.** `relbench`'s metadata pins
`scikit-learn<=1.6.1`; this venv has 1.7.2 and other experiments depend on it
(`sentence-transformers`). The only sklearn APIs relbench touches are
`roc_auc_score`/`average_precision_score`/`f1_score`/`mean_absolute_error`/`r2_score`,
all stable across that gap. `pandas`, `pyarrow`, `datasets`, `numpy` are already present.

Then add to `requirements.in` with a comment recording the `--no-deps` reason, and
re-`pip-compile`. Verify the metric functions import and run on a toy array before
building anything.

### 3.2 `pyg-lib` — the sampling backend (§2.1)

```bash
.venv/bin/pip install pyg-lib \
    -f https://data.pyg.org/whl/torch-2.11.0+cu130.html
```

Resolves to `pyg_lib-0.8.0+pt211cu130-cp310-abi3-manylinux_2_28_x86_64.whl`. The install is
**additive** — a package this venv does not currently have — so it cannot disturb `torch`,
`transformers`, or the already-broken `torch-sparse`/`torch-scatter`. Do **not** reinstall
those two as part of this experiment; `pyg-lib` alone satisfies `NeighborSampler`, and
touching them changes packages other experiments import.

Two things to check before depending on it:

1. **glibc.** The wheel is `manylinux_2_28`. Confirm `ldd --version` ≥ 2.28 on the login
   node *and* inside the training container
   (`/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh`) — the
   container is where sbatch runs actually execute, and it is a different userspace.
2. **It actually flips the flag.** The acceptance check:

   ```bash
   .venv/bin/python -c "
   import torch_geometric.typing as T
   assert T.WITH_PYG_LIB, 'pyg-lib still not detected'
   print('WITH_PYG_LIB', T.WITH_PYG_LIB)"
   ```

Pin `pyg-lib==0.8.0` in `requirements.in` with the `-f` index URL in a comment, since it is
not on PyPI for this torch/CUDA combination.

### 3.3 Acceptance gate for phase 0

A single script, `check_env.py`, that must pass before §4 begins:

* `relbench` imports; `relbench.metrics.roc_auc` runs on a toy array under sklearn 1.7.2;
* `WITH_PYG_LIB` is `True`;
* `NeighborLoader` performs a temporal sample on `relbench.datasets.fake.FakeDataset`
  converted to `HeteroData` (§5.1), and **every sampled node satisfies `time <= seed_ts`**.

That last item is §5.3 test 1 run against a toy graph, before any real data exists. If it
fails, fall back to the hand-rolled sampler described in §5.2's appendix and reinstate the
original ~200-line estimate.

**Status: implemented as `check_env.py`, all 7 checks passing (2026-07-27.)** Three findings
worth carrying forward, all of which would have cost time later:

1. **`subgraph_type="induced"` does not work on heterogeneous graphs.** `pyg-lib` has not
   implemented it (`neighbor_sampler.py:440` guards its branch with
   `subgraph_type != SubgraphType.induced`), so an `induced` request falls through to the
   `torch-sparse` branch — the broken one — and raises
   `"requires either 'pyg-lib' or 'torch-sparse'"` *even with pyg-lib correctly installed*.
   The error names the wrong cause; budget for that if anything similar appears later. Use
   **`subgraph_type="directional"`**, which is what we want regardless: it keeps the edges
   the sampler actually traversed (the fkey path from the seed) rather than an arbitrary
   induced closure.
2. **PyG's temporal filter is `time <= seed`, and that is correct**, not merely tolerable.
   relbench builds its label window as
   `re.date > t.timestamp AND re.date <= t.timestamp + timedelta` (`tasks/f1.py:95-96`) —
   strictly *after* the seed — so a row stamped exactly at the prediction instant is past
   context, not label. It is also what RDL sees, since `relbench/modeling/loader.py` drives
   the same sampler. `check_env.py` asserts the inclusive boundary so a future PyG change
   to `<` cannot silently shrink every neighborhood.
3. **Hop attribution works.** `batch[table].num_sampled_nodes` gives the per-hop counts and
   `n_id` maps back to source rows, so `(table, row, hop)` triples (§5.2) are recoverable
   with no bookkeeping of our own.
4. **`temporal_strategy="last"` returns the right set in ASCENDING time order.** Found while
   testing the sampler, not by reading the docs. It selects the k most recent eligible rows
   — correct — but hands them back oldest-first, so taking a prefix of the raw output keeps
   the *oldest* rows of the retained window. The allocator must re-sort by time descending.
   This is the worst class of bug in this experiment: the graph stays valid, stays leak-free,
   and is simply and silently worse. `check_env.py`'s 3b sorted before comparing and so did
   not catch it; `test_recency_order_under_last_strategy` does.

The `STATIC_TIME` sentinel is `-(2**40)`, not `iinfo(int64).min`: far before any real
timestamp, but with headroom if anything downstream takes a difference of two times.

**Cache location.** `relbench` caches under `pooch.os_cache("relbench")` unless
`RELBENCH_CACHE_DIR` is set. Pin it to a repo-local, gitignored path so login node and
compute nodes agree:

```
export RELBENCH_CACHE_DIR=src/experiments/relbench/raw_data
```

Set it in `config.py` (`os.environ.setdefault`) so it holds regardless of shell, and add
`src/experiments/relbench/raw_data/` + `processed_data/` to `.gitignore`.

---

## 4. Phase 1 — download and inspect (login node, no GPU)

```python
from relbench.datasets import get_dataset, download_dataset
from relbench.tasks import get_task
db   = get_dataset("rel-f1", download=True).get_db()      # 9 tables, pandas DataFrames
task = get_task("rel-f1", "driver-dnf", download=True)
train_t, val_t, test_t = (task.get_table(s) for s in ("train", "val", "test"))
```

`download_dataset` pulls `db.zip` from `relbench.stanford.edu` (pooch, hash-checked) —
**this must happen on the login node**; compute nodes have no outbound internet. Task
tables are computed locally by duckdb from the db and cached under the same dir.

Deliverable of this phase — an `analyse_dataset.py` (kgqa has the pattern) that prints and
saves `dataset_stats.json`:

* per-table: rows, columns, dtypes, null fractions, whether it has a `time_col`, its
  fkeys, and the p50/p95/max string length of every text-ish column;
* per-task: `task.stats()` (rows per split, class balance for the binary tasks, target
  quantiles for the regression one), the split timestamps, `timedelta`,
  `num_eval_timestamps`;
* the **degree distribution of `drivers` → each child table**, which is what actually sets
  the neighborhood budget (how many `results`/`qualifying`/`standings` rows a driver
  accumulates before a given date).

Nothing downstream is designed until these numbers exist. In particular `max_nodes` and
the fanout are chosen *from* the degree distribution, not guessed.

### 4.1 Measured results (2026-07-27) — five corrections to this plan

`analyse_dataset.py` is written and run; full output in `analysis/rel-f1_stats.json`.

**1. `driver-dnf` is *not* balanced, and its base rate drifts hard across the split.**

| split | rows | positive rate |
|---|---:|---:|
| train | 11,411 | **0.880** |
| val | 566 | 0.779 |
| test | 702 | **0.705** |

§12.2 called it "50/50 balance" and picked it as the debugging task partly for that. The
balance claim was wrong. It is still the right debugging task (most rows, cheapest eval,
least contaminated), but two consequences follow: an always-yes baseline scores **70.5%
test accuracy**, so accuracy is nearly uninformative and AUROC/AP must carry the reporting;
and a LoRA fine-tune will absorb an 88% positive prior from train and meet a 70% one at
test. Log the predicted-positive rate per eval (already a §11 row) and watch it against
these numbers, not against 50%. `driver-top3` is stable by comparison (0.171 / 0.202 /
0.176).

**2. The neighborhood is 3–7× larger than the planned `max_nodes` sweep.** Temporally
filtered hop-1 eligible rows, over 500 sampled train seeds:

| task | p50 | p90 | p99 |
|---|---:|---:|---:|
| `driver-dnf` | 53 | 239 | 393 |
| `driver-position` | 48 | 229 | 361 |
| `driver-top3` | 111 | 339 | 463 |

`max_nodes ∈ {16, 32, 64}` (§5.4) would discard most of the eligible context for the median
seed and nearly all of it at p90. **Sweep `{32, 64, 128}` instead**, and treat 128 as the
likely operating point. At ~60–80 chars per row that is roughly 3k tokens — comfortable for
the backbone, and the O(N²) SPD/magnetic features are still cheap at N=128. Note this also
puts us well *above* the baseline's document sizes (211–4,314 tokens), which strengthens
the token-redundancy comparison of §5.4 fix 4.

**3. `qualifying` is empty for most seeds.** Zero eligible rows for **84%** of
`driver-position` seeds and **82%** of `driver-dnf` seeds — the table only starts in 1994
while train spans 1950–2004. So §5.4's budget allocator step 3 (redistribute the unused
share of a starved relation) is the *common* path, not an edge case; it must be correct and
tested, not treated as a refinement. `driver-top3` is the exception at 7% zero, because its
train split starts in 1994 too.

**4. The docstring rule needs a second discriminator.** §5.0 A's fallback assumed "has a
docstring or not". There is a third case: `results-position` **is** `AutoCompleteTask`
itself — a generic per-column factory, not a subclass — so it *has* a docstring, and that
docstring reads *"Args: dataset: The dataset object..."*. Splicing it into a question is
worse than having none. The rule is now: use the docstring only when
`type(task).__module__` starts with `relbench.tasks.` (concrete tasks) rather than
`relbench.base.` (framework). Implemented as `_task_docstring()` in `analyse_dataset.py`
and to be shared verbatim with the question template.

**5. The entity is not always a dimension row.** `results-position` and
`qualifying-position` seed on *fact* tables: 0 child relations, 3 parent relations
(`raceId`, `driverId`, `constructorId`). Any code that counts only children reports an
empty neighborhood. Both are phase-1 non-goals, but the sampler and the allocator must
handle upward-only neighborhoods or §5.0 A's zero-code-change claim fails on the first
dataset with a fact-table entity task.

### 4.2 `rel-trial` measured (2026-07-27) — a different shape of neighborhood

Downloaded, graph built, surveyed over 300 `study-outcome` train seeds. `analysis/
rel-trial_stats.json`. The entity table `studies` is a hub with **10 child relations**
(rel-f1's `drivers` has 3), four of them junction tables.

**1. Four of the ten relations are permanently empty — correctly so.**
`outcomes`, `outcome_analyses`, `drop_withdrawals` and `reported_event_totals` are at
**100% zero** for every seed. They are stamped with the trial's *completion* date, not its
start, so for any given study they always fall after the seed. That is the temporal filter
working exactly as intended: those tables hold the trial's **results**, and
`reported_event_totals` literally contains the adverse-event counts `study-adverse`
predicts. Seeing 100% zero here is evidence of correctness, not of a broken sampler — worth
stating plainly so nobody later "fixes" it.

**2. What remains is width, not depth.**

| relation | mean | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|
| `facilities_studies` | 34.8 | 3 | 85 | 538 | 999 |
| `conditions_studies` | 2.0 | 2 | 4 | 6 | 8 |
| `sponsors_studies` | 1.5 | 1 | 2 | 5 | 9 |
| `interventions_studies` | 1.2 | 1 | 3 | 7 | 11 |
| `designs` | 1.0 | 1 | 1 | 1 | 1 |
| `eligibilities` | 1.0 | 1 | 1 | 1 | 1 |
| **total hop-1** | **41.5** | **11** | **94** | **542** | |

rel-f1's neighborhood is *depth* — dozens of rows of the same three relations, a driver's
race history. rel-trial's is *width* — one `designs` row, one `eligibilities` row, a couple
of conditions and interventions, each from a different table with a different schema, plus
a long `facilities` tail. The signal is concentrated in ~8 information-dense nodes: the seed
`studies` row (29 columns including `detailed_descriptions`, `brief_summaries`,
`official_title`), `designs` (15 columns) and `eligibilities` (15 columns, criteria text).

Three consequences:

* **`max_nodes` should be *small* on rel-trial and `max_node_chars` large** — the opposite
  of rel-f1's setting. Sweep ~{12, 24, 48} rather than {32, 64, 128}. This is the
  text-vs-nodes tension of §1.1 risk 1, now measured rather than predicted.
* **The budget allocator is load-bearing, not a refinement.** `facilities_studies` runs to
  p99 538 and max 999 while contributing the least per row (hospital name, city, country).
  Without the even-split-across-relations rule it takes the entire budget and starves
  `eligibilities` and `designs`, which is where the signal is.
* **Heterogeneity is the structure here.** Seven distinct table schemas around one seed is
  genuinely graph-shaped in a way a driver's 50 near-identical `results` rows is not, and it
  is exactly what a flat JSON document expresses worst.

**3. `label_history` does not exist on rel-trial — and that is an advantage.**
`study-outcome` has **11,994 rows over 11,994 distinct studies: 1.00 rows per entity.**
Compare `driver-dnf`: 11,411 rows over 780 drivers, **14.63 per entity.**

So the §5.4 `label_history` axis is structurally empty here: there are no past task-table
rows for the same entity, because each study is seeded exactly once. §5.4 called label
history "the one input we are otherwise missing" and §8.7 made it an ablation precisely
because on rel-f1 it is so predictive it can mask whether the database is read at all — the
baseline's best 1B `driver-dnf` config used label history and **zero** database rows.

On rel-trial that confound is gone by construction: **the relational neighborhood is the
only signal available.** That makes it a far cleaner test of whether GTLM actually reads
relational context, and it is an argument for the retarget (§1.1) I did not have when the
decision was made. Keep `label_history` implemented — it applies to rel-f1 and to any
recurring-entity task — but expect it to resolve to 0 on rel-trial, and make §8.7's
ablation rel-f1-only.

**Also worth noting, not yet acted on:** the >95%-null column rule keeps
`results.milliseconds` (78.5% null), `results.fastestLap` and `results.rank` (both 89.7%
null) — three mostly-empty columns spending tokens on `nan`-omissions. And the
"cardinality == row count ⇒ free-form id" rule flags `circuits.name`, `constructors.name`
and `circuits.lat/lng`, which is wrong: those are real signal *and* precisely the
memorizable content §8.6's contamination arm exists to control rather than silently delete.
Both thresholds need revisiting at §6.1, with the document dump as the evidence.

**Leakage note to write into the module docstring:** always use
`get_db(upto_test_timestamp=True)` (the default). The per-seed temporal filter (§5) is what
makes train/val correct; `upto_test_timestamp` is the coarse second fence. Never call
`get_db(upto_test_timestamp=False)`.

For test-set metrics we need unmasked labels: `task.get_table("test", mask_input_cols=False)`.
That is legitimate for internal reporting (the labels ship in the download), and it is the
call `task.evaluate()` itself makes by default.

---

## 5. Phase 2 — the relational index and the temporal neighborhood sampler

Two modules, both pure numpy/pandas, both testable without a GPU or a model.

### 5.0 Governing constraint: the *method* is task- and dataset-agnostic

The constraint is on the **method**, not on the hyperparameter values. Two different things
were conflated in an earlier draft of this plan; keeping them apart matters.

#### (A) Hard constraint — genericity of the code

**No human writes anything task-specific or dataset-specific.** Adding a new task or a new
database must require *zero* code changes: point the config at
`(dataset, task)` and the pipeline runs. Concretely, everything that is necessarily
specific to a task is **derived from `relbench` metadata by one rule**:

| Necessarily specific | Forbidden | Derived instead from |
|---|---|---|
| The question text | a hand-tuned prompt per task | one template over `task.entity_table`, `target_col`, `task_type`, `timedelta`, and — *when present* — the task's docstring (`Task.__doc__`) |
| The answer verbalization | per-task label words | `task_type`: binary → `yes`/`no`, regression → `%.2f`, multiclass → the class value's string |
| Regression bucket edges (if `bucket` readout) | hand-picked ranges | train-split quantiles, computed at prep time |
| Which columns a row contributes | a curated per-table list | auto-derivation: drop pkey/fkey, drop >95% null, drop columns whose cardinality equals the row count (free-form ids), keep the rest |
| Which columns get aggregated | a per-task feature list | every numeric column of the child table, plus count and recency — mechanically |
| How the node budget is spread over relations | per-dataset tuning | the budget allocator (§5.4), which reads only the schema |

`COLUMN_SPEC` stays in the codebase as an escape hatch for a pathological column (a
free-text blob that alone blows the token budget), but using it is recorded in the run
record so a reader can see the method was overridden. Headline runs use the auto-derived
spec. **A test asserts that a fresh `(dataset, task)` pair resolves end-to-end with no
entry in any hand-written dict.** That test is the operational definition of this
constraint.

**`Task.__doc__` is not universally available — verified 2026-07-27.** Every task class in
`amazon/arxiv/avito/event/f1/hm/mimic/ratebeer/stack/tgb/trial.py` carries a one-line
`r"""..."""` description, but **all 14 in `tasks/dbinfer.py` have none**. So the question
template must degrade gracefully: with a docstring it is used verbatim as the task
description; without one, the template falls back to `entity_table` + `target_col` +
`task_type` + `timedelta` alone, and the run record logs which branch fired. **The
genericity test should hold out a `dbinfer` task specifically** — it is the adversarial
case, and a test that only ever sees documented tasks would pass while the rule is broken.

#### (B) Orthogonal choice — how hyperparameter *values* get set

Per-task hyperparameter selection on validation does **not** violate (A): no human inspects
the task, the procedure is identical everywhere, and a new task gets its values
automatically. It is also what every baseline here does — RDL, LightGBM and the LLM
baseline all tune per task (§5.0.1).

So do not pay the +5.7 AUROC that a single frozen construction costs. Instead, since the
construction sweep produces both numbers from the *same runs*, **report both**:

* **`global`** — the single construction with the best mean rank across tasks. The
  "one setting for everything" number; the stricter, foundation-model-shaped claim.
* **`per_task`** — the construction with the best validation metric for each task
  independently. Comparable to the baseline's per-task-tuned headline.

Both must be computed **identically for every arm** (graph, flat, no-bias) — the selection
procedure is part of the protocol, and applying it to one arm only would manufacture a
result. Selection touches validation only; test is scored once at the end, per §7.3.

The corollary for §5.4's menu: `include_siblings` and `aggregates` are swept, and their
*mean* effect is what decides the `global` construction, while `per_task` may land on
different values for different tasks. The per-task breakdown is the interesting mechanism
story either way — record it.

#### 5.0.1 What actually gets trained — one model per task

For the record, since it sets the norm we are matching:

| System | What is trained | Granularity |
|---|---|---|
| LightGBM (RelBench baseline) | the GBT | **one per task** |
| RDL (RelBench baseline) | GNN + task head, end-to-end | **one per task** |
| Wydmuch et al., LLM columns | *nothing* — the backbone is frozen | one frozen Llama for everything |
| Wydmuch et al., `+ MLP` columns | a 10-unit MLP on one token embedding, ≤1e5 docs | **one head per task** |
| **This experiment** | LoRA + graph-bias params on a frozen backbone | **one per task** (3 for rel-f1) |

So there is no rel-f1-level model and no RelBench-level model anywhere in the literature we
are comparing to; the only cross-task sharing is a frozen backbone. One GTLM per task is
the standard protocol, and it is what phase 1 does.

**Optional later arm — a genuinely multi-task model.** Training one GTLM on all three
rel-f1 tasks jointly is natural for us and awkward for RDL (different heads, different label
spaces), because our targets are all *text* and the task is stated in the QUESTION node.
That would be a real differentiator — "one model, three tasks, no per-task head" — and the
machinery costs almost nothing: concatenate the built datasets and let the question text
disambiguate. Do it only after the per-task headline exists, so there is something to
compare against, and report it as a bonus claim rather than the main result.

### 5.1 `graph_build.py` — relbench `Database` → `HeteroData`, cached once per dataset

`relbench.modeling.graph.make_pkey_fkey_graph` does this already, but it builds
`torch_frame.TensorFrame` features for every column and therefore drags in `torch_frame`.
We want the topology only — the *columns* become text later (§6.1), never tensors. So:
same graph, ~100 lines, no feature encoding.

For every table:

* `pkey` is guaranteed consecutive `0..n-1` by relbench (`validate_and_correct_db`
  asserts it), so a row index *is* the node id within its node type — no hash maps, and
  `n_id` maps straight back to a pandas row.
* `data[table].time`: `int64` unix seconds from `time_col`. Static dimension tables
  (`drivers`, `circuits`, `constructors` in rel-f1) have no `time_col` and get
  `torch.iinfo(torch.int64).min`, so they always satisfy `time <= seed_ts` and are
  permanently eligible. (Sentinel choice matters: `0` would also work for rel-f1 but breaks
  on any database with pre-1970 timestamps.)
* For each fkey `(child_table, fkey_col) → parent_table`, one edge type
  `(child, f"f2p_{fkey_col}", parent)` with `edge_index` from the fkey column, dropping
  rows where it is null (relbench already nulls dangling fkeys).
* `T.ToUndirected()` so the sampler can traverse parent→child as well; the *stored*
  direction for the GTLM graph is re-derived at §5.2's output step, not taken from here.

Cache to `processed_data/<dataset>/graph.pt` keyed by a `db_version` string. Arrays stay
`int32` where they fit and load with `mmap` for the large datasets (§10).

### 5.2 `sampler.py` — `NeighborLoader` wiring + the policies PyG lacks

**Default path — PyG does the sampling.** One `NeighborLoader` per split:

```python
NeighborLoader(
    data,                                   # the HeteroData from §5.1
    num_neighbors=fanout,                   # per-relation, per-hop — RDL's own knob
    input_nodes=(task.entity_table, seed_rows),
    input_time=seed_timestamps,             # the task's prediction timestamps
    time_attr="time",
    temporal_strategy="last",               # == §5.4's `recent`; "uniform" == `uniform`
    subgraph_type="directional",            # NOT "induced" — see §3.3 finding 1
    batch_size=1, shuffle=False,
)
```

`input_time` is the load-bearing argument: PyG enforces `neighbor.time <= seed.time` along
the whole sampled path, per seed, which is exactly relbench's temporal semantics. This is
node-level time, so it does **not** hit the `WITH_EDGE_TIME_NEIGHBOR_SAMPLE` guard at
`neighbor_sampler.py:325`.

Our wrapper is thin and does four things PyG does not:

* **Map back to rows.** `batch[table].n_id` gives the original row indices; the per-hop
  split comes from `batch[table].num_sampled_nodes` (with
  `num_sampled_nodes_per_hop` when `subgraph_type` allows). Emit
  `nodes: list[(table, row, hop)]`.
* **Re-orient the edges.** Output directed **child → parent** (fact → dimension), undoing
  `ToUndirected`. Direction is not cosmetic — the magnetic Laplacian is the only bias
  channel that carries edge direction (per the `direction` probe), and child→parent is the
  semantically meaningful orientation.
* **Global budget** `max_nodes`: `num_neighbors` bounds fanout per relation but not the
  total, so the allocator trims by (hop ascending, time descending). Two corrections found
  by running it, both recorded in §4.2 / §3.3 finding 4:
  * **The budget buys *content* rows, not rows.** Link-table rows (§`graph_build.link_tables`
    — every column a pkey, fkey or timestamp) ride along free, because they are topology
    with no information: a `facilities_studies` row is an id pair and a date while the
    facility's name is one hop further out. Billing for them starves the dimension tables
    on any junction schema.
  * **Candidates are pooled across hops**, not filled hop-1-first as originally written.
    On rel-trial the content sits at hop 2 behind a hop-1 junction row, so a hop-ordered
    budget is exhausted before reaching it.
* **Determinism.** PyG's sampler draws from the global torch RNG; seed it per example with
  `torch.manual_seed(stable_hash(f"{dataset}:{task}:{split}:{entity}:{ts}:{version}"))`
  before each `loader` draw, so a rebuild reproduces byte-identical graphs and the
  `samples_per_node` version index is stable across machines. `temporal_strategy="last"` is
  deterministic anyway; this only binds for `uniform` and `mixed`.

**Output contract, unchanged:** `(nodes: list[(table, row, hop)], edges: list[(i, j)])` and
nothing else. No text, no tokens — so everything downstream is unit-testable against
hand-built fake databases regardless of which sampler produced the nodes.

#### What still needs hand-written sampling

Three of §5.4's five policies are not expressible through `NeighborLoader`, and all three
are optional arms rather than the critical path:

| policy | why PyG can't | how we do it |
|---|---|---|
| `recent_plus_strided` | needs log-spaced offsets deep into history, not a contiguous recent slice | sample with an inflated `num_neighbors`, then reselect by recency index. Cheap on rel-f1; revisit for the large datasets |
| `paper_match` | needs a *table-level* visited set and denormalized parent inlining — deliberately non-graph semantics | its own function; it exists to reproduce a document format, not to sample a graph |
| `mixed` | half-recent/half-uniform in one draw | two `NeighborLoader` draws (`last` + `uniform`), union, trim |

`include_siblings` turns out to need **no** special handling: child → parent → other
children is plain 2-hop sampling on the undirected graph, so it falls out of `fanout` having
a second entry. `aggregates` is post-processing over the full temporally-valid set (§5.4),
not a sampling policy at all.

*Appendix — the fallback.* If §3.3's acceptance gate fails (`temporal_strategy="last"`
misbehaving on hetero graphs, or the wheel not loading in the container), revert to a
hand-rolled BFS: reverse-CSR per fkey via `order = np.lexsort((-child_time, fkey_vals))` +
`offsets = np.searchsorted(...)`, making "k most recent children before T" a slice. ~200
lines, and it keeps the same output contract, so nothing downstream changes.

### 5.3 The tests that must exist before anything is built on top

These test **our wrapper's contract**, not PyG's internals — but test 1 deliberately
re-verifies the temporal filter end to end anyway. Trusting a dependency is not the same as
assuming it; the cost of the assertion is negligible and the cost of it being wrong is the
whole experiment.

1. **No future rows.** For 1000 random seeds across all three splits: every sampled row's
   `time` is the static sentinel or `<= ts`. This is the single most important test in the
   experiment — a bug here makes the task trivial and the result worthless. A cheap version
   of it also runs in §3.3's phase-0 gate, before any real data exists.
2. **Fanout/budget respected**; `len(nodes) <= max_nodes` always.
3. **Determinism**: same seed → identical node list; different `version` → different but
   reproducible.
4. **Reachability**: the seed row is node 0 and every node is connected to it.
5. Run 1–4 against `relbench.datasets.fake.FakeDataset`, so CI needs no download.

### 5.4 Selection policies — matching the LLM baseline, then beating it

#### What the baseline actually does (Algorithm 1, arXiv:2411.11829v1)

Their document for a query entity *X* at seed time *t<sub>p</sub>* is:

```
[database description + task description]              # prose, from the RelBench paper
[n_inc  in-context examples]  each = a train-table row (X', y') with t_X' < t_p,
                              expanded, serialized as JSON *with its label*
                              — stratified for binary tasks, and the SAME set is
                                reused for every document in a task
[n_rel  related examples]     each = the most recent train-table rows *for the same
                              entity*, t < t_p, expanded, JSON *with its label*
[the query entity X]          expanded, JSON, target field last
```

and `ADD_RELATED_ENTITIES` expands one row by:

1. following **every** fkey → pkey link, unconditionally, and inlining the parent row;
2. then, for each pkey → fkey (one-to-many) link, taking the **`n_nest` most recent**
   children with `t < t_X`, and recursing to depth `d`;
3. maintaining a **table-level visited set** — once a table has been expanded anywhere in
   this document, it is never expanded again.

Grid: `n_inc ∈ {0,8,16}`, `n_rel ∈ {0,8,16}`, `n_nest ∈ {0,4,8}`, `d ∈ {0,1}`, best config
picked per task on validation. Average rel-f1 document lengths (their Table 7, Llama-3.2
tokenizer): 211 tokens at `(0,0,d0,nest0)`, 655 at their best `driver-dnf` config, 914 at
their best `driver-top3` config, up to 4,314 at the full `(16,16,d1,nest8)` corner.

#### The one input we are otherwise missing: label history

`n_rel` is not a database neighborhood — it is **the entity's own past task-table rows,
with their labels**. For `driver-dnf` that is "did this driver DNF in each of the previous
16 windows", which is enormously predictive and is why their best 1B config uses `n_rel=16`
and *no database rows at all*. RDL does not get this by default; that asymmetry is most of
their rel-f1 margin over RDL.

It is temporally legitimate (all rows satisfy `t < t_p`), so we should use it — but
**explicitly, as its own axis**, not smuggled in:

* `label_history: 0 | 8 | 16` — the k most recent train-table rows for the same entity,
  each becoming **one node** whose text is `"<date>: <target_col> = <value>"`, edged to the
  seed row.
* It must be available to the flat control on identical terms, or the architecture
  comparison is corrupted.
* Train-split subtlety: when building a *train* example, its own row must be excluded from
  its own history (`X'' ∉ D_X` in their pseudocode does this by row identity). Assert it.

#### Where their strategy is weak, and what I would do instead

Four weaknesses, each with a concrete fix. All four are cheap for us and impossible or
expensive for a JSON-tree document:

Each fix is a **schema-level rule** — it reads only the table structure, the timestamps and
the column dtypes, so the *code* applies unchanged to every task and every database (§5.0 A).
Its on/off value may still be chosen per task on validation (§5.0 B); what may never happen
is a human deciding "on rel-f1 we also pull X".

1. **Recency-only truncation.** `n_nest` most recent means a driver's last 4–8 races —
   roughly half a season. Entity-level labels generally depend on *long-run behaviour* at
   least as much as current activity, and their own numbers show `d=1` traversal *hurting*,
   which is what an under-informative recent slice looks like.
   → **`recent_plus_strided`**: take `k_recent` most recent **plus** `k_hist` rows
   log-spaced over the entity's whole eligible history (offsets 1, 2, 4, 8, 16, 32 …
   rows back). Same node budget, spans years instead of months, and the rule is pure
   recency-index arithmetic — nothing task-specific in it. I expect this to be the single
   biggest accuracy lever in the experiment, ahead of anything on the bias channel.
2. **No aggregates.** A list of k rows forces the model to compute counts and means
   in-context, which 1B models do badly — and it is exactly what LightGBM gets by hand and
   RDL gets from message passing. → **aggregate summary nodes**: one synthetic node per
   (seed entity, child relation) carrying deterministic aggregates over **all** temporally
   valid rows — count, mean/median/min/max of **every numeric column** (chosen by dtype, not
   by hand), time since last row, and counts over trailing 90/365-day windows. One node
   summarizes an unbounded number of rows, it is a pure function of legal data, and it is
   available to both arms. `aggregates: off | seed | all` as a global axis.
3. **The table-level visited set forbids returning to a table by a second path.** Because
   `results` is marked visited after the seed entity's own results are expanded, their
   document can never contain *another* row of `results` reached via a different parent —
   on rel-f1, another driver's result in the same race. This is a generic structural
   limitation: in any schema, a fact table joins two or more dimensions, and the *co-rows
   sharing a parent* are the peer group the seed is being compared against.
   → **`include_siblings`**: for each sampled fact row, pull up to `sibling_fanout` co-rows
   sharing its parent, temporally filtered like everything else. The rule is
   "child → parent → other children", stated over the schema, with no knowledge of what the
   task asks. It is genuinely graph-shaped context, it is what a JSON tree can only express
   by duplicating rows, and it is the sharpest architectural argument available on this
   benchmark.
   *Prediction (recorded now, to be checked later):* peer context should help where the
   label is comparative (`driver-top3` — qualifying in the top 3 is a contest) and be
   roughly neutral where it is intrinsic (`driver-dnf` — a reliability question about one
   car). If that asymmetry appears, the `per_task` selection will pick it up automatically
   and the `global` setting will follow the mean — which is exactly the pair of numbers
   §5.0 B asks for, and the contrast between them *is* the mechanism evidence.
4. **Denormalization duplicates parents.** Inlining the `races` row inside every `results`
   row repeats it k times; our Levi-style graph names each row once and lets the topology
   carry the join. → No fix needed, it is free for us — but **measure it**: report the
   flat/graph token-redundancy ratio exactly as kgqa did (it found 1.99× on WebQSP). On
   rel-f1 with `n_nest=8`, a `races` row repeated 8× should push this well above 2×, which
   is a token-efficiency result worth reporting in its own right.

So `neighbor_sampling` is the axis, with these values:

| value | rule | why it exists |
|---|---|---|
| `paper_match` | recency-only, `d=1`, table-level visited set, denormalized parents | the like-for-like input condition for quoting their number |
| `recent` | recency-only, node-level visited set, per-relation fanout | our clean baseline |
| `uniform` | uniform over eligible | RDL's policy; the control for "is recency doing the work" |
| `recent_plus_strided` | k recent + k log-spaced | fix (1) — expected default after the ablation |
| `mixed` | half recent, half uniform | makes `samples_per_node > 1` a real augmentation |

with `aggregates` and `include_siblings` as orthogonal booleans. The construction ablation
sweeps `neighbor_sampling × aggregates × include_siblings × label_history` across **all
three rel-f1 tasks**, which yields both selections at once (§5.0 B): the `global`
construction by mean rank, and the `per_task` construction by each task's own validation
metric. Sweeping all three tasks rather than the cheapest one is what makes the `global`
number meaningful — selecting it on a single task would be per-task tuning wearing a
disguise.

#### The budget allocator (a rule, so `max_nodes` stays a single number)

`max_nodes` has to mean something on both a 9-table sports schema and a 15-table clinical
one. Rather than hand-tuning the split per dataset, allocate it mechanically:

1. Reserve slots for the seed row, the `QUESTION`/`PROMPT` nodes, `label_history`, and one
   aggregate node per child relation.
2. Split the remainder **evenly across the seed's child relations**, so no relation starves
   because another has millions of rows.
3. Redistribute unused slots (a relation with fewer eligible rows than its share) round-robin
   to relations that are still saturated.
4. Only then fill hop 2 with what is left.

This is scale-free: a wide schema gets fewer rows per relation, a narrow one gets more, and
no config edit is involved. `max_nodes` itself is swept (32/64/128 — sized from the measured degree distribution,
§4.1 finding 2) as part of the
construction selection, and — like every other construction knob — resolves to a `global`
value and a `per_task` value (§5.0 B).

**Split-size note.** rel-f1 task tables are small. Measured from the actual download
(2026-07-27): `driver-dnf` 11,411/566/702, `driver-top3` **1,353**/588/726,
`driver-position` 7,453/499/760. The first two match the paper's Table 4 exactly; their
`driver-position` reads 7,533/499/864, so **their numbers come from a different relbench
version than the one we run**. Immaterial for the two classification tasks, but any
`driver-position` MAE quoted against their 3.539/3.092 is not strictly like-for-like and
should say so.
`driver-top3` having ~1.3k train rows is a real constraint: a LoRA fine-tune will overfit
it in a few hundred steps, so it needs `samples_per_node ≥ 4` augmentation, frequent
`eval_steps`, and its seed variance reported honestly. It is also why their ICL approach
looked so good there. Note these are *training-schedule* accommodations, set by a rule from
the split size rather than by inspecting the task — §5.0 A is about the code being generic,
not about every task receiving numerically identical settings. Keep that line clean:
schedule knobs may vary with split size,
sampling and serialization knobs may not.

---

## 6. Phase 3 — rows to text, and the graph

### 6.1 `row_text.py`

One node = one row, rendered as

```
results | date: 1998-03-29 (312 days before prediction) | grid: 4 | position: 2 |
points: 6.0 | laps: 56 | status: Finished
```

Knobs, all part of the cache key:

* `text_mode`: `key_value` (above, default) | `sentence` (per-table natural-language
  templates) | `compact` (table name + values only). Cheap ablation, and `sentence` is the
  arm that should benefit most from a pretrained backbone.
* **Column selection** — a per-dataset `COLUMN_SPEC` dict with an auto-derived default:
  drop `pkey_col`, drop every fkey column, drop columns that are >95% null. **Dropping the
  fkey columns is deliberate**: the edges carry that information structurally, and leaving
  raw ids in the text both wastes tokens and invites id memorization — which on a temporal
  split is pure overfitting. The flat control (§8) gets the *same* dropped columns, so the
  comparison stays matched.
* **`time_encoding`: `absolute` | `relative` | `both` (default `relative`).** Rendering
  every timestamp as an offset from the seed timestamp is what RDL gets for free and a
  serialized LLM does not. Absolute dates additionally let the model memorize eras, which
  a temporal split punishes. Worth a 3-arm ablation early — I expect this to be the
  rel-f1 equivalent of kgqa's data-format v3.
* **`anonymize`: `none` | `entities` | `all`** — the contamination control (§8.6). Under
  `entities`, the name columns of dimension tables are replaced by a stable per-row token
  (`driver_314`) rather than deleted, so node count and topology are unchanged and only the
  memorizable content moves. Under `all`, `time_encoding` is additionally forced to
  `relative`.
* **Nulls are omitted, not printed as `nan`** — absence is a cleaner signal and shorter.
* Floats rounded (`%.4g`), long strings truncated at `max_node_chars`.
* Foreign-key *names* are recovered structurally: the driver's name lives on the `drivers`
  node, which is in the subgraph, so `results` doesn't need to repeat it.

### 6.1.0 `text_mode` — three document strategies (added 2026-07-27)

`key_value` repeats the column name on every row. The alternative is to hoist a table's
column list into a **header node** and render its rows as bare positional values, wiring each
row to its header with a `row -> header` edge. Three modes, `key_value` the default:

| mode | what it does |
|---|---|
| `key_value` | labels every field on every row; drops fields a row does not populate. The control |
| `schema_node` | hoists **every** sampled table into a header node |
| `shortest` | hoists a table only when that is actually shorter for the rows this graph sampled |

**Why a positional row must pad.** `key_value` can drop a null field because its neighbours
are self-labelling. A positional row cannot: the header promises column *i* at slot *i*, so a
dropped field shifts every later value into the wrong column — silently, since the document
still reads fine. Aligned rows therefore emit `NULL_SLOT` (`-`) for every unpopulated column,
and that padding is what makes hoisting expensive on a sparse schema.

**Measured** (120 train graphs each, full document including question and prompt nodes):

| | `key_value` | `schema_node` | `shortest` |
|---|---:|---:|---:|
| rel-f1 `driver-dnf`, n=64 | 1,803 tok | 1,216 (−32.5%) | **1,210 (−32.9%)** |
| rel-trial `study-outcome`, n=24 | 630 tok | 712 (**+13.0%**) | **567 (−10.0%)** |

Two things decide it, and they point opposite ways on the two datasets:

1. **Row multiplicity.** A header is paid once per *table*; key names are saved once per
   *row*. rel-f1 has three tables at ~15 rows each (`races` 16.0, `standings` 15.9,
   `results` 14.7); rel-trial has eight at ~1.85 rows each, so it pays the schema eight
   times to save key names twice.
2. **Fill rate.** A header declares the whole schema; a labelled row only pays for the
   columns it fills. rel-f1 is 92.0% filled (39 kept columns, 35.9 populated per row);
   rel-trial is 68.9% (82 kept, 56.5 populated — `designs` 39.0%, `outcome_analyses` 50.9%,
   `studies` 66.5%).

`shortest` compares both renderings in characters on the rows actually sampled, so it adapts
to both without a threshold to tune and **cannot be longer than `key_value`** — verified on
240/240 graphs. It hoisted 3.55 tables/graph on rel-f1 and 1.91 on rel-trial.

**One asymmetry between the arms, under the hoisting modes only.** The graph arm appends
header nodes *after* the rows (its prefix is bidirectional, so position is irrelevant); the
flat arm must emit them *before* (it is read causally, and a header after its rows is
useless). So the two arms carry identical lines in identical row order but place the header
lines differently. Under the default `key_value` there are no header lines and the arms stay
byte-identical, which is the condition the headline comparison runs under.

### 6.1.1 Measured at the document-dump gate (2026-07-27)

`row_text.py` and `dump_documents.py` are written and the documents have been read. Four
things changed as a result; all were invisible until a human looked at the text.

**1. `pandas.isna()` does not see relbench's nulls.** rel-trial encodes missing values as the
literal string `"None"`: `studies.is_ppsd` and `studies.fdaaa801_violation` are **100%**
`"None"`, `is_unapproved_device` 99.5%, `baseline_type_units_analyzed` 99.9%. Every one of
them looked fully populated to the null filter and would have rendered in every document
forever. rel-f1 has the same problem in MySQL dialect — `drivers.code` is `\N` in 88.3% of
rows. Thirteen columns are now dropped from `studies` alone.

The fix is **column-level, not value-level**, because `"None"` is genuinely ambiguous:
`designs.masking: "None"` (20.4% of rows) means *open-label*, a real category. So placeholder
strings count toward a column's missing fraction — a column that is 100% `"None"` carries
nothing on either reading — but are still rendered where the column survives. Only
unambiguous markers (`\N`, `<NA>`, `NaT`, empty) are dropped per value.

**2. Pure join rows were 6 of 19 nodes.** A `conditions_studies` row renders as
`conditions_studies | date: 158d before` — no content, and a date it already shares with the
study. `collapse_links` (default on, ablatable) contracts them into direct edges, exactly as
kgqa collapses unnamed CVT mediators. A rel-trial document goes 19 → 15 nodes with no
information lost, and `condition -> study` is a better edge than `condition -> junction ->
study`. Orientation is higher-hop → lower-hop, which keeps the convention that edges point
back toward the seed.

**3. Cosmetics that are not cosmetic.** Postgres booleans arrive as `t`/`f` (`has_dmc`,
`adult`, `subject_masked`) — one character a language model has no prior over, now rendered
`true`/`false`. `eligibilities.criteria` uses `~` as its line separator, which reads as noise
mid-sentence; now `; `.

**4. We are giving the model *less* context than the baseline, not more.** Measured document
sizes, Llama-3.2-1B tokenizer:

| | nodes p50 | tokens p50 | tokens p90 | tokens/node |
|---|---:|---:|---:|---:|
| rel-f1 `driver-dnf`, `max_nodes=32` | 31 | 1,108 | 1,188 | 36.2 |
| rel-trial `study-outcome`, `max_nodes=24` | 11 | 555 | 980 | 47.5 |

The baseline's rel-trial documents run **1,571–42,720** tokens. At 555 we are a factor of
three below their *smallest* configuration, mostly because the free text in
`detailed_descriptions` and `criteria` is being truncated — which is precisely what the whole
retarget (§1.1) is about.

> **Correction (2026-07-27): the culprit was `max_node_chars`, not `max_value_chars`, and the
> fix recorded here originally would have done nothing.** Measured on the full tables at
> `max_value_chars=200`: rel-trial `studies` rows are 866 characters untruncated, so
> `max_node_chars=600` cut **95.5%** of them — and `studies` is the seed node, the one holding
> `detailed_descriptions` and `brief_summaries`. `outcomes` was cut 38.3%, `outcome_analyses`
> 13.5%. Raising `max_value_chars` to 1200 behind a 600-character node cap changed the total
> document length by ~0 tokens, which is exactly what was observed and misread. rel-f1 was
> never affected (longest row: 179 characters), so the cap only ever bound where it did the
> most damage. `max_node_chars` now defaults to `None` (no node cap): `max_value_chars`
> bounds each field and `max_length` bounds the sequence, so a node is already bounded twice.
> `validate()` rejects `max_node_chars < max_value_chars` as incoherent, and the document dump
> reports the truncation rate so this cannot recur silently.

§5.4's budget sweep should carry `max_value_chars` as an axis alongside `max_nodes`; on a
text-heavy database it is probably the more important of the two — and now it actually
reaches the text.

**Also measured:** rel-f1 produces a **seed-only graph** (no eligible history at all) for 5%
of `driver-dnf` train rows — a driver whose first prediction date precedes any of their
races. Those examples have no evidence to reason from and should be counted in the analysis,
not silently averaged in. rel-trial: 0%.

### 6.2 `data.py` — graph assembly, mirroring kgqa/tag_benchmarks

Per (entity, timestamp, version) build an `nx.DiGraph`:

* the sampled database rows (§5.2), plus — when enabled — the `label_history`,
  aggregate-summary and sibling nodes of §5.4. History and aggregate nodes edge to the seed
  row; sibling nodes edge to their shared parent, exactly as a real fact row does.

* one node per sampled row, `text=row_text(...)`, seed row prefixed `TARGET `;
* directed child→parent edges from the sampler;
* a **`QUESTION` node** (`question_node: isolated` default) holding the task instruction +
  the entity's identity + the prediction date and window, e.g.

  > Prediction date 2005-03-06. Over the next 30 days, will driver #22 fail to finish
  > (DNF) at least one race?

  `isolated` because it won the kgqa ablation outright and edge-mode made no difference
  there; the bidirectional prefix mask already exposes it to every graph token, which is
  what makes the *encoding* question-conditioned. Keep `off` / `seed` as arms.
* a **`PROMPT` node**, `graph.graph["prompt_node"] = "PROMPT"`, text `"Answer: yes"` /
  `"Answer: no"` / `"Answer: 4.50"`, with `AnswerLabelMasker`-style masking on the **last**
  occurrence of `"Answer:"` so only the label token(s) are supervised.
* `graph.graph` also stores `target` (raw float/int), `entity`, `timestamp`, `split`,
  `row_id` — the evaluator needs to line predictions back up with `task.get_table(...)`
  **in table order**, since `task.evaluate` compares positionally.

Then, exactly as tag_benchmarks does: `TextGraphDataset(graphs, per_graph_versions=...)`
→ `tokenize` → `compute_shortest_path_distances` → `compute_magnetic_lap` →
(`compute_rrwp` only if enabled — it was 13× the storage for nothing on WebQSP, so default
it **off** here) → `compute_labels` → `save` in `CHUNK_SIZE` chunks under
`processed_data/<data_config_key>/{train,val,test}/`.

**Ordering constraint:** build in the exact row order of `task.get_table(split)` and never
shuffle the val/test datasets, so prediction vector index *i* ↔ table row *i*. Assert it
(store `row_id` and check monotonicity at load).

---

## 7. Phase 4 — scoring: the new eval path

This is where RelBench diverges from every existing experiment in the repo.

### 7.1 Binary classification → AUROC / AP

Target is a single token (`" yes"` / `" no"` — **verify single-token-ness with the Llama
tokenizer first**; if either splits, fall back to `" A"`/`" B"`).

Score = **`logit(yes) − logit(no)` at the answer position, computed in fp32.** Not the
softmax probability: the margin is strictly monotone in it, avoids bf16 saturation ties,
and AUROC/AP only care about ranking. Ties are the failure mode to watch — log the number
of distinct scores per eval.

Implementation: a `preprocess_logits_for_metrics` that gathers *two* logits per example
(never letting HF buffer `(B, L, V)`), and a `compute_metrics` that calls **relbench's own
`roc_auc` / `average_precision` / `f1` / `accuracy`** so the reported numbers are literally
the benchmark's implementations. One forward pass, no generation — this eval is *cheaper*
than kgqa's.

`metric_for_best_model = "eval_roc_auc"`, `greater_is_better=True`.

**Two requirements from the §1.2 audit. Both are silent if missed.**

1. **Apply `sigmoid` to the margin before `task.evaluate`.** `gnn_entity.py` does
   `pred = torch.sigmoid(pred)` for binary tasks, and it matters: `driver-dnf`'s metric list
   is `[average_precision, accuracy, f1, roc_auc]`, and `relbench.metrics.f1` /
   `.accuracy` threshold at `pred >= 0.5`. On a raw unbounded logit margin those two are
   meaningless — a model predicting "no" everywhere with margin −3 still scores `f1` as if
   it predicted all-positive. `roc_auc` and `average_precision` are rank-based and unchanged
   by the monotone transform, so the headline number is unaffected either way; the point is
   that the *other three reported metrics* are garbage without it. Sigmoid also maps the
   margin to the baselines' own quantity: they score with `P(token "1")` (arXiv:2411.11829
   §3.2, "metric-aware inference"), and `sigmoid(logit(yes) − logit(no))` is the two-way
   renormalization of exactly that.
2. **Build the target table from the stored `row_id`, not by assuming identity.** With
   `max_val_samples` set, the val cache is strided, so `task.evaluate(pred,
   task.get_table("val"))` sees a length mismatch. Index the target instead —
   `Table(df=target.df.iloc[row_ids], ...)` — which makes a capped val split evaluate
   correctly and keeps the full-split path unchanged. Test is never strided
   (`assert_row_order(..., contiguous=True)` enforces it), so test always uses the full
   table and `task.evaluate(pred)` with no target argument, exactly as the baselines do.

### 7.1.1 M6 smoke, measured (2026-07-27)

`002_smoke.jsonc`: `driver-dnf`, `max_nodes=16`, 200 train rows, **8 optimizer steps**, both
arms, H100. The AUROCs below carry no information about the model — 8 steps at batch 1 ×
accum 4 is 32 examples seen — but three properties were confirmed on real data:

| arm | test AUROC | val | distinct scores | runtime |
|---|---:|---:|---:|---:|
| base | 0.393 | 0.695 | 33/702 | 15.9 s |
| flat | 0.496 | 0.560 | 8/702 | 10.5 s |

1. **The alignment holds.** `evaluate_split` recomputes each metric twice — once from the
   gathered answer token, once through `task.evaluate` against the task table — and raises on
   disagreement. It did not raise, on the full 702-row test table, for either arm. The guard
   is separately proven able to fail: `test_evaluate_crosscheck.py` feeds it a shuffled cache
   (self-consistent by our bookkeeping, wrong by relbench's positional comparison) and
   requires the raise.
2. **The sigmoid is right.** At step 8 the model predicts all-positive and `f1 = 0.8768`,
   which is exactly `2p/(1+p)` for the observed `pos_rate = 0.7807`. That identity holds only
   if the threshold is applied to a probability rather than to the raw margin.
3. **Tie collapse is real and must be watched.** 8 distinct scores across 702 test examples
   on the flat arm. The arithmetic points at bf16 quantization rather than a bug: at logit
   magnitude ~10 the bf16 ulp is 0.03–0.06, so a saturated model whose margins span ~0.2 can
   only produce a handful of values, and an 8-step model *is* saturated. `n_distinct` is in
   every run record. **If it stays low once a trained run spreads the margins, AUROC is being
   compressed by ties rather than by the model**, and the fix is to compute the two logits
   from the final hidden state in fp32 (the bf16 rounding happens inside the `lm_head`
   matmul, so casting its output afterwards recovers nothing).

### 7.2 Regression → MAE / RMSE / R²

Two readouts; implement `numeric_text` first, keep `bucket` as the fallback:

Note the baseline paper found its distributional readout ("median of `P(y|x)` by sampling
probabilities of values between the train min and max") *performed poorly* — which is why
their regression table reports only the MLP-head variant. Our fine-tuned setting is
different (the model is trained to emit the number), but it is a warning that the readout,
not the model, can be the bottleneck. Hence the parse-failure rate below is a gate, and
the `bucket` arm exists.

* **`numeric_text`** — greedy `model.generate` of ≤ 8 tokens from the truncated prompt
  node, parse the first float, fall back to the **train-split median** on parse failure
  (and log the parse-failure rate as a first-class metric — if it is >1% the readout is
  broken, not the model). Slim version of `kgqa/evaluate.py`'s generative loop; reuse its
  flex→eager switch during generation.
* **`bucket`** — quantize the target into K quantile buckets at prep time, supervise the
  bucket label, predict `Σ p_k · centroid_k` from one forward. No parse failures, gives a
  calibrated distribution, costs a discretization bias. Worth running as a second arm on
  `driver-position` regardless.

### 7.3 Reporting

A `RelBenchTrainer(GraphTrainerV2)` that overrides `evaluate` to attach these metrics
(exactly the shape of `KGQAGraphTrainer`), plus a final `evaluate.py` pass that:

* rebuilds the full prediction vector in table order,
* calls `task.evaluate(pred, task.get_table(split, mask_input_cols=False))` — so **every
  metric in the task's own `metrics` list** lands in the JSONL,
* writes `predictions.npy` next to the run for post-hoc analysis.

---

## 8. Phase 5 — controls, in the same PR as the main arm

Non-negotiable, and the lesson of the kgqa thread: build the control before you believe
the number.

1. **Flat text control** (`arm: flat`) — the *same sampled rows*, same order, serialized as
   lines into a single node, no structural bias, same base model and schedule. Copy
   `kgqa/flat_data.py` + `flat_train.py`. Byte-identical supervision. This is control #1
   and the primary scientific comparison.
2. **No-bias graph control** (`spd/rrwp/magnetic` all off) — separates "one node per row +
   per-node RoPE reset + bidirectional prefix" from "the structural bias channel". This is
   the arm tag_benchmarks' `003_textonly_ablation` runs.
3. **Label-shuffle canary** — train with targets permuted within split; AUROC must land at
   0.5. Catches leakage in the sampler, the label masker, and the score readout at once.
4. **Leakage canary (deliberate)** — one run with the temporal filter disabled. It should
   score conspicuously *better*; if it doesn't, the neighborhood carries no temporal signal
   and the whole sampling design needs revisiting. Run once, never merge as a default.
5. **Majority/median baseline** and **LightGBM-on-flattened-features** are already in the
   RelBench papers — cite, don't rebuild.
6. **Contamination control (`anonymize`)** — the one rel-f1 specifically demands, given the
   baseline paper's own admission that its models lean on memorized Formula-1 knowledge.
   Three arms, identical in every other respect:
   * `none` — real names (`forename`, `surname`, constructor and circuit names).
   * `entities` — drop or hash the name columns of the *dimension* tables, keeping every
     fact row intact. The model must then work from the relational history alone.
   * `all` — additionally strip absolute dates (`time_encoding: relative`), removing the
     era cue.

   The delta `none − entities` **is** the memorization estimate, and it is the number that
   makes `driver-top3` interpretable. Report it beside every rel-f1 headline. If the gap is
   large for the flat control and small for GTLM, that is a real finding about where each
   arm's signal comes from; if it is large for both, the honest conclusion is that rel-f1
   is a weak benchmark for relational reasoning and the text-heavy databases (§10) matter
   more than adding rel-f1 tasks.
7. **`label_history` ablation** (`0 / 8 / 16`) — because the baseline's best 1B `driver-dnf`
   config uses *only* label history and no database rows, an arm at `label_history=0` with
   database rows on is the clean test of whether our neighborhood is contributing anything
   at all. If `label_history=16, database rows off` matches our full arm, we have
   reproduced their result and learned that the database is not being used — which would be
   a finding, not a failure, and would redirect the whole thread toward §10.

---

## 9. Phase 6 — plumbing, tests, and the sweep

Files to create under `src/experiments/relbench/` (sizes are estimates):

| File | ~LoC | Responsibility |
|---|---:|---|
| `config.py` | 350 | `RunConfig` (every knob), `validate()`, `bias_params()`, `lora_config()`, `data_config_key()`, task-type helpers, `COLUMN_SPEC`/`INSTRUCTIONS` per dataset |
| `check_env.py` | 60 | phase-0 acceptance gate (§3.3): relbench + `WITH_PYG_LIB` + a temporal sample on `FakeDataset` |
| `graph_build.py` | 100 | relbench `Database` → `HeteroData` (topology + time only, no `torch_frame`) |
| `sampler.py` | 120 | `NeighborLoader` wiring, row/hop mapping, edge re-orientation, budget trim, determinism |
| `sampler_policies.py` | 120 | the three policies PyG lacks: `recent_plus_strided`, `paper_match`, `mixed` |
| `row_text.py` | 150 | row → node text; column specs; time encoding |
| `data.py` | 300 | `run_data_prep_mode(cfg)` / `load_data(cfg)`; chunked `.gtds` cache |
| `flat_data.py` | 150 | the flat serialization control, from the same sampler output |
| `evaluate.py` | 250 | score readouts, `RelBenchTrainer`, `task.evaluate` reporting |
| `train.py` | 250 | one config → one JSONL record (tag_benchmarks' shape) |
| `flat_train.py` | 200 | control arm's trainer |
| `__main__.py` | 300 | argparse ↔ `RunConfig`, `--init`, mode dispatch |
| `test.py` | 100 | re-score a checkpoint |
| `analyse_dataset.py` | 200 | phase-1 stats + budget-sizing report |
| `_io.py`, `__init__.py` | 30 | `append_jsonl` |
| `README.md` | — | reproduce-from-scratch, as kgqa's |

Tests (`tests/experiments/relbench/`): `test_sampler_temporal.py` (the five §5.3 tests),
`test_row_text.py`, `test_label_masking.py`, `test_score_readout.py` (synthetic logits →
known AUROC), `test_relbench_flags.py` (the sweep render→parse→config round-trip that every
experiment here pins).

`data_config_key()` — the cache identity — must include: `dataset`, `task`, `hops`,
`fanout`, `max_nodes`, `neighbor_sampling`, `label_history`, `aggregates`,
`include_siblings`, `sibling_fanout`, `anonymize`, `text_mode`, `time_encoding`,
`column_spec` version, `max_node_chars`, `question_node`, `model_name` (tokenizer!),
`max_length`, `magnetic_q`, `rrwp`+`max_rw_steps` (only when on, so caches stay valid when
it's off — kgqa's convention), and `data_seed`. Training-only knobs (seed, lr, k_hop, bias
arms) must **not** be in it, so ablation arms share one built dataset.

Sweep configs, in order:

* `001_data_prep.jsonc` — build every cache the later configs reference (CPU, no GPU).
* `002_smoke.jsonc` — `--max-steps 8`, tiny splits, both task types, local mode. Proves
  the whole path end to end in minutes.
* `003_f1_canary.jsonc` — label-shuffle + leakage canaries.
* `004_construction_ablation.jsonc` — **runs before the headline and fixes the
  construction(s) used from then on.** `neighbor_sampling` ∈ {`paper_match`, `recent`,
  `recent_plus_strided`} × `aggregates` ∈ {off, seed} × `include_siblings` ∈ {false, true}
  × `label_history` ∈ {0, 16} × **all 3 tasks**, 1 seed = 72 runs, then the surviving
  constructions re-run at 3 seeds. One sweep, two selections (§5.0 B): `global` by mean
  rank across tasks, `per_task` by each task's own validation metric — both recorded, both
  carried into the headline. The neighborhood is the dominant variable on this benchmark
  (the baseline's grid swings ±20 AUROC across document parameters), so fixing it before
  comparing architectures is the only ordering that yields an interpretable headline.
  `max_nodes` ∈ {32, 64, 128} rides along as a final pass on the surviving constructions (§4.1).
* `005_f1_headline.jsonc` — 3 tasks × {graph, flat, no-bias} × 3 seeds = 27 runs at the
  `global` construction, plus the same grid at each task's `per_task` construction where it
  differs, `per_config` sbatch, B200/B300, `max_concurrent` 8. **The selection procedure is
  applied identically to all three arms** — selecting per task for the graph arm and
  globally for the flat control would fabricate the result we are trying to measure.
* `006_paper_match.jsonc` — the `paper_match` construction × {graph, flat} × 3 tasks, so
  the external comparison is quotable at matched inputs.
* `007_contamination.jsonc` — `anonymize` ∈ {none, entities, all} × {graph, flat} × 3 seeds
  on both classification tasks.

---

## 10. Phase 7 — generalizing to the other subsets

The pipeline is dataset-agnostic by construction; per new dataset the only work is:

1. a `COLUMN_SPEC` entry (or accept the auto-derived default),
2. an `INSTRUCTIONS` entry per task,
3. budget re-sizing from `analyse_dataset.py`'s degree distribution.

Two things will bite at scale and should be designed for now, not retrofitted:

* **Graph memory.** `rel-amazon`/`rel-stack`/`rel-trial` have tens of millions of rows.
  Keep `edge_index` `int32` where it fits, and rely on `NeighborLoader`'s `share_memory` /
  worker path rather than materializing per-seed subgraphs in the parent process. This is
  a second reason to have taken the PyG route (§2.1): its sampler is already the one tuned
  for graphs at this scale. rel-f1 will not exercise it; build for it anyway.
* **Prep wall-clock.** Per-seed sampling is O(fanout × hops × log n); the cost is dominated
  by tokenization and the O(N²) SPD/magnetic features. At `max_nodes ≈ 32` that is
  trivial, but a dataset with 100k+ train rows wants the chunked build to be resumable
  (tag_benchmarks' `_is_built` per split is already most of the way there).

**This section has been overtaken by §1.1.** `rel-trial` is no longer the follow-up; it is
the headline, for the reasons recorded there (its rows carry paragraphs, and a frozen 1B
scores 55.72 on `study-outcome` against 88.47 on rel-f1's `driver-top3`). What remains
genuinely later is `rel-stack` (840 MB, post bodies) and `rel-amazon` (6.1 GB, product
descriptions and reviews) — the same argument at a scale where prep cost is the binding
constraint. Download sizes measured 2026-07-27: rel-f1 0.7 MB, rel-event 100 MB, rel-hm
136 MB, rel-avito 348 MB, rel-trial 548 MB, rel-stack 840 MB, rel-amazon 6.1 GB.

---

## 11. Risks, ranked, and how each is checked

| Risk | Why it matters | Check |
|---|---|---|
| **Temporal leakage** | Makes the task trivial; result is worthless and the error is silent | §5.3 test 1 on every split + the label-shuffle canary + the deliberate leakage arm |
| **Prediction ↔ table row misalignment** | `task.evaluate` compares positionally; a shuffle silently destroys the metric | Store `row_id`, assert monotonic on load, never shuffle val/test |
| **Score ties in bf16** | AUROC collapses toward 0.5 for reasons unrelated to the model | fp32 logit margin; log distinct-score count per eval |
| **Class imbalance** | Measured: `driver-dnf` is 88% positive on train. A model that always says yes is not obviously broken | Report AUROC/AP as primary (as RelBench does); log the predicted-positive rate against §4.1's per-split base rates, not against 50% |
| **Base-rate drift across the temporal split** | `driver-dnf` runs 0.880 train → 0.779 val → 0.705 test. A fine-tune absorbs the train prior and meets a different one at test; val selection partly compensates, test does not | Log per-split predicted-positive rate; prefer AUROC (prior-invariant) for selection; treat any accuracy/F1 delta smaller than the 17pp base-rate shift as uninterpretable |
| **Regression parse failures** | Silently replaced by the median → flattering MAE | Parse-failure rate is a first-class logged metric; >1% fails the run |
| **Neighborhood too small to be informative** | GTLM and flat both bottleneck on retrieval, and the comparison measures nothing | Size `max_nodes`/fanout from the measured degree distribution; run a budget sweep (16/32/64) before the headline |
| **`--no-deps` install drifts** | A future relbench version needs a dep we skipped | Pin `relbench==2.1.2` in `requirements.in` with the reason in a comment |
| **`pyg-lib` wheel does not load in the training container** | The wheel is `manylinux_2_28` and abi3; the container is a different userspace from the login node, and sbatch runs execute *there* | §3.2 checks `ldd --version` in both; §3.3's gate is run inside the container too, not just on the login node |
| **PyG's temporal semantics differ subtly from relbench's** | `input_time` enforcing `<=` vs `<`, or hetero `temporal_strategy="last"` behaving differently from the homogeneous case, would leak the label window silently | §3.3 gate on `FakeDataset` with hand-checkable timestamps, *before* wiring; then §5.3 test 1 at scale. Fallback sampler kept in §5.2's appendix |
| **Static-table sentinel** | Using `0` for tables with no `time_col` breaks on any database with pre-1970 timestamps | `torch.iinfo(torch.int64).min` (§5.1); assert no real timestamp equals the sentinel |
| **Compute nodes have no internet** | Data prep on sbatch fails at `download_dataset` | Download on the login node in phase 1; `data_prep` asserts the cache exists and errors with the exact command to run |
| **Pretraining contamination on rel-f1** | `driver-top3` at 88.47 AUROC is Llama knowing F1, not relational reasoning; a GTLM win could be the same thing | The `anonymize` arm (§8.6) is mandatory on every rel-f1 headline |
| **`driver-top3` has 1,353 train rows** | A LoRA fine-tune overfits it in a few hundred steps; seed variance will be large | `samples_per_node ≥ 4`, frequent `eval_steps`, 3 seeds minimum, report the spread not just the mean |
| **The neighborhood dominates the architecture** | The baseline's own grid swings ±20 AUROC across document parameters — larger than any plausible graph-vs-flat effect | Run `004_construction_ablation` **before** the headline and fix the construction |
| **Method genericity erodes silently** | A hand-written column list or prompt tweak for one stubborn task turns "one method" into per-task engineering, and the headline stops meaning what it says | A test resolves a held-out `(dataset, task)` pair end-to-end with no entry in any hand-written dict; every run logs whether `COLUMN_SPEC` was used |
| **Asymmetric selection between arms** | Selecting per task for the graph arm and globally for the flat control would manufacture a win | One selection procedure, applied to every arm, recorded in the run record |
| **Selection-task bias in the `global` number** | Choosing the global construction on one task and applying it everywhere is per-task tuning wearing a disguise | Select `global` by mean rank over ≥3 tasks spanning both task types (§5.0 B) |

---

## 12. Open decisions (recommendation first)

0. **Task-agnostic *method* (§5.0 A)** — settled. No task- or
   dataset-specific code, prompts, column lists or budgets; a new `(dataset, task)` pair
   runs with zero code changes, and a test enforces it. **Hyperparameter values are a
   separate question (§5.0 B):** the construction sweep yields both a `global` setting and
   a `per_task` setting from the same runs, and both are reported for every arm. This keeps
   the strict claim available without paying the +5.7 AUROC that per-task tuning is worth
   to the baseline.
0b. **One model per task** (§5.0.1) — 3 models for rel-f1, matching RDL, LightGBM and the
   baseline's MLP heads. A jointly-trained multi-task GTLM is an attractive *later* arm and
   a genuine differentiator (our targets are text, so no per-task head is needed), but it is
   not the phase-1 headline.
0c. **Sampling backend** (§2.1, §3.2) — settled: install `pyg-lib` and use PyG's
   `NeighborLoader`. The earlier "write our own" decision rested on a false premise
   (`temporal_strategy="last"` *does* express most-recent-k) and an overstated one (the
   missing backend is a published wheel away). The hand-rolled BFS survives only as §5.2's
   fallback appendix, gated on §3.3.
1. **Regression readout** — start with `numeric_text`, add `bucket` as a second arm on
   `driver-position`. If parse failures exceed 1%, `bucket` becomes the default. (The
   baseline paper's distributional readout failed outright, so do not assume this is free.)
   Note this is a *readout* choice, not a construction choice — it may legitimately differ
   between task types, since it is dictated by the metric, not tuned for accuracy.
2. **Which task to debug on** — `driver-dnf` (11,411 train rows, cheapest eval path, least
   contaminated; **88% positive on train, not balanced** — §4.1 finding 1). But **construction selection runs on all three** — the
   debugging task and the tuning task are deliberately not the same thing.
3. **Ship the flat control in phase 1** — yes. The kgqa thread's most valuable single
   artifact is its flat control; retrofitting it costs a re-run of everything.
4. **Include `label_history`** — yes, as an explicit axis available to both arms, default
   16. Excluding it would leave us strictly below the baseline's input condition and make
   the comparison meaningless; hiding it inside "the neighborhood" would make it
   unattributable.
5. **In-context examples (`n_inc`)** — **skip them.** They are a substitute for fine-tuning,
   and we fine-tune. Their only role would be reproducing the baseline's `driver-top3`
   config, which is the contaminated one anyway. Revisit only if `006_paper_match` needs it
   for a like-for-like quote.
6. **`k_hop` masking** — leave at 0. It has hurt almost everywhere it was tested, and the
   subgraph here is 2 hops wide by construction.
7. **RRWP** — off by default (13× storage for −0.8 F1 on WebQSP). Revisit only if
   SPD+magnetic underperforms the flat control.
8. **`include_siblings`** — my strongest bet for where GTLM beats a serialized document,
   and the one input their table-level visited set structurally forbids. Ship it globally
   if its mean effect is positive. The predicted per-task asymmetry (helps the comparative
   label, neutral on the intrinsic one) is recorded as a *falsifiable mechanism claim* to be
   checked in the analysis — not as a reason to enable it per task.
