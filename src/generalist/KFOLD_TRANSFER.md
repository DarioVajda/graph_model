# The k-fold transfer study

Plan of record for the adaptation experiment that replaces `temp.gtds/TODO.md`
§9.2 and §9.3. Written 2026-09-21.

## 1. What this measures, and why the old design could not

The claim the generalist campaign wants to make is *breadth buys adaptation*:
a model pretrained on a broad molecular mixture learns a **new** molecular task
in fewer steps than the backbone does.

§9.2 as written cannot test that, because all three of its tasks — BACE, BBBP,
HIV — are in the trunk's own mixture. Measured directly: the trunk of record
already scores 0.8152 on BACE test and 0.7558 on HIV test before an adapt leg
trains a single step, against a pre-registered 95 % bar of 0.7792 and 0.7306.
The trunk starts *above* the target, `steps_to_target` is 0 or one eval, and the
ratio is an artifact of `eval_steps` rather than a measurement. Any metric of the
form "how much specialist training is the trunk worth" on an in-mixture task is
really asking whether multi-task training trains the task. It does. That is not
transfer.

§9.3 does test the right thing, but only on the three tasks that happen to be
permanently held out, and it cannot say whether the result generalises.

**The fix is to hold out tasks on purpose, in folds.** Train one trunk per fold
with that fold's tasks removed from the mixture, then measure adaptation on them.
Every task gets a turn at being novel. This is task-level k-fold
cross-validation; grouping by family rather than at random is the standard
refinement, and it is the one that matters here.

## 2. Clustered folds, not stratified

Two ways to build the folds, and they measure different things.

*Stratified* — each fold holds out one ring task, one group task, one property —
measures **near transfer**: learning a new task when siblings are still in the
trunk. It carries a confound with no answer: hold out `ring_size` while
`ring_count`, `ring_membership` and `aromatic_ring` remain and any reader says
"of course it adapts fast, you left three near-twins in there."

*Clustered* — each fold holds out an entire family — measures **far transfer**:
learning a genuinely novel capability with no sibling to lean on. Its weakness
is different in kind: each fold's trunk is missing a whole capability, so none of
them is exactly the model we would ship. That is a caveat, not a refutation, and
it goes in a sentence.

Clustered, therefore. It is also the harder test, so a result that survives it is
worth more than one that survives the easy version.

## 3. The folds

The three formerly permanent held-outs — `bond_path`, `longest_chain`,
`clintox` — **fold in like any other task.** A permanent holdout existed because
the single-trunk design needed *something* to measure adaptation on; a k-fold
design manufactures held-out tasks on demand, so keeping them out of every trunk
costs their training signal four times over and buys nothing that three trunk
seeds do not already give.

| fold | theme | tasks | n |
|---|---|---|---:|
| **A** | topology | `ring_membership`, `aromatic_ring`, `ring_size`, `ring_count`, `bond_path`, `longest_chain` | 6 |
| **B** | groups and stereo | `fg_presence`, `fg_count`, `fg_atom_membership`, `stereo_potential`, `stereo_assigned` | 5 |
| **C** | properties | `bace`, `bbbp`, `hiv`, `tox21`, `sider`, `clintox` | 6 |
| **D** | generation | `g2s`, `chebi20` | 2 |

Nineteen tasks, no remainder. B bundles two small families rather than one, which
is fine: the holdout logic only requires that no task's siblings are left behind,
and fg and stereo are each removed whole. A takes the path/distance tasks with
the cycle tasks for the same reason — they are not twins, but close enough that
leaving one kind in would put the sibling confound back.

**A and B are the clean folds; C and D are not, and the write-up must say so.**
`BLOCKS` water-fills a removed task's share *inside its own block*, so:

- A removes 6 of 11 Tier-A families; the block holds 25 % and the surviving five
  go 2.27 % → 5.0 % each. Every other block untouched.
- B removes 5; the surviving six go 2.27 % → 4.17 %. Tier-A still 25 %.
- C removes the **whole** Tier-B block. 40 % has nobody inside it to absorb the
  share, so it spills across the others and Tier-A/ChEBI/g2s go from 25/20/15 to
  roughly 42/33/25.
- D removes two whole blocks, 35 %, and Tier-B goes 40 % → ~62 %.

So C and D are not "the generalist minus a cluster", they are structurally
different models. Their comparison against a fresh baseline is still internally
valid — the fold's tasks really are absent from the fold's trunk — but they
answer "what does this block contribute" rather than "does breadth help", and
they must not be pooled with A and B.

## 4. Budget

**Trunk: 11,140 steps, `tokens_per_step` 16,384, `budget_scale` 2.0 — identical
to `008`, for every fold.** Holding total tokens fixed is what keeps trunk size
out of the comparison. Each fold's trunk then gives more passes to its surviving
tasks; that is the water-fill doing its job and it is unavoidable.

**The pass caps have to be re-derived per fold, and getting this wrong kills the
run at 90 %.** Tier-A families and `g2s` carry no `passes` key. An uncapped task
that runs out does not retire quietly — it asks for a pass `data_prep` never
built and raises `AdapterBuildError`. Fold A roughly doubles each surviving
Tier-A family's share, so it consumes built passes twice as fast. Before any
submission: compute passes-consumed per surviving task at the fold's share and
confirm the build materialised that many. `tier_a_cap_per_pass` is 4,000.

**Adapt legs: size so the *from-base* leg plateaus with margin.** On BACE it
plateaued by step 250–300 of 1,000, so 1,000 is generous there; a structural task
learned from scratch needs more. Start at 2,000 for Tier-A and topology, 1,000
for Tier-B, and confirm on fold A.

**Evals must be log-spaced.** Crossings land in the first few hundred steps,
where a flat `eval_steps: 50` gives almost no resolution, and the resolution of
the grid is the resolution of every number this study reports.

## 5. Schedule

**10-step warmup, then constant.** Horizon-free, which is the whole point: with
cosine-to-a-horizon a leg that would cross later is penalised by a decaying rate,
so the crossing step becomes partly an artifact of the budget chosen, and
changing the budget later invalidates every number already collected. Constant LR
means the curve at step N does not depend on `max_steps`, and a leg can be
extended without redoing anything.

`lr` 1e-4, settled by the §9.1 screen: from-base end score on BACE was val 0.6777
against 0.6666 at 3e-5 and test 0.8403 against 0.8279, so both splits pick the
same rate.

The cost is a noisier plateau and a slightly worse final score than a decayed
schedule. Acceptable when the quantity is crossing time. For a clean asymptote,
fork a short cooldown off the end of a leg — the same shape as trunk → anneal.

Both legs must share `lr`, `warmup_steps`, `tokens_per_step` and a fresh
optimizer. One asymmetry stays and gets named rather than fixed: the from-parent
leg starts with a trained LoRA adapter and the from-base leg with a fresh one.
Part of what pretraining buys is that the adapter is not random, so it belongs in
the number, but a reader will ask.

## 6. Measurement

- **Read the crossing on the test split.** This is measurement, not selection:
  the reported quantity *is* the crossing step, and the threshold is a constant
  from a prior campaign, so nothing about this experiment's test data chooses
  anything. Test is also the quieter signal — mean absolute eval-to-eval change
  on the BACE screen leg was 0.0111 on test against 0.0162 on val. `fork.py`
  refuses a test-named target under D7.4; adapt mode needs a documented
  exemption, not a workaround.
- **Require three consecutive crossings.** `steps_to_target` returns the *first*
  crossing, and a first-passage time over a noisy signal is optimistically
  biased — it picks the step where the metric happened to look good. With 152
  test molecules on BACE and ~0.011 jitter, one crossing means little. Three
  consecutive costs 100 steps of reporting lag at `eval_steps` 50, which is the
  right trade and gets stated.
- **Threshold: 95 % of the specialist score, on the split it is read on.** The
  existing 0.8133 / 0.6882 / 0.7336 are the specialists' *test* numbers and the
  fork names them on *val*; BACE's val is harder than its test and BBBP's is far
  easier, so the mismatch changes sign per corpus and no offset repairs it. Read
  on test, the specialist test numbers are the right constants. Note that 95 % of
  a metric with a 0.5 floor is not 95 % of the way there — 0.95 × 0.8202 = 0.7792
  is 87 % of the range above chance — so state which convention is used.
- **Report the full curves and an area-under-curve number.** Time-to-threshold's
  known weakness is sensitivity to an arbitrary threshold; the transfer
  literature pairs it with AUC for exactly that reason. AUC is free once the
  curves exist.

### 6a. The thresholds

**One convention for all fourteen reported tasks: 95 % of the mean annealed
score of the folds that trained the task.** 95 % of the score, not of the range
above chance; for an AUROC that is the weaker statement and it is the one being
made.

| task | metric | threshold | source |
|---|---|---|---|
| nine in-mixture Tier-A families | `em_accuracy` | cross-fold | the annealed runs of every fold that trained it |
| `mol/bace`, `mol/bbbp`, `mol/hiv` | `roc_auc` | cross-fold | as above; the specialist rides along as a comparison, below |
| `mol/chebi20` | `bleu2` | cross-fold | as above |
| `mol/g2s` | `roundtrip_match` | cross-fold | as above |
| `mol/bond_path`, `mol/longest_chain` | `em_accuracy` | self | the base leg's own final score — **appendix only**, see below |
| `mol/tox21`, `mol/sider` | `roc_auc` | — | gradient tasks, never an anchor number (§1) |

**BACE, BBBP and HIV do not anchor on their specialists**, though they are the
only three tasks that have one and their forks were submitted carrying 95 % of
it. Three rows of a fourteen-row table on a strictly harder bar than the other
eleven is not a table: a ratio measured against one bar does not compare with a
ratio measured against another, and the harder bar also risks both legs failing
to reach it inside a 1,000-step budget, which returns nothing at all. The
specialist number is the better comparison than target — it is the only
measurement in the study of how far below a dedicated model the generalist level
sits, which is exactly the limitation the cross-fold convention carries and
cannot otherwise quantify. It is reported in the `spec95` column and anchors
nothing.

| task | specialist (test) | 95 % of it | source |
|---|---:|---:|---|
| `mol/bace` | 0.8202 | 0.7792 | `molecules/TODO.md` §6, 3 seeds, sd 0.0120 |
| `mol/bbbp` | 0.7056 | 0.6703 | same, sd 0.0229 |
| `mol/hiv` | 0.7691 | 0.7306 | same, sd 0.0172 |

Changing the bar cost nothing and needed no resubmission: a target is read once,
after both legs have run, off a recorded curve. That is the same property §6a
relies on for the deferred anchors.

**The cross-fold mean is over folds, with seeds averaged within.** Fold A runs
three seeds and B, C and D one apiece, so a flat mean over cells would give fold
A three fifths of the weight in every non-A anchor and pull the threshold toward
fold A's mixture — a property of how the pilot was sized, not of the task.

**bond_path and longest_chain are out of the headline.** Their `self` anchor
answers a different question — "how much sooner does the trunk reach what the
backbone ends at" rather than "how much sooner does it reach competence" — and a
second convention costs more to explain than two rows are worth. They keep
running and keep scoring; they print below the table, labelled `self`, and they
stay out of anything quoted as the study's result. The six extra forks that
existed only to give them four parents each (B, C and D, parent leg alone) were
cancelled on 2026-09-22, and fold A's own two were not resubmitted after the
validator crash below — so as of that date the two tasks have no leg running.
`legs()` still emits them, and the trunk checkpoints are on disk, so a leg goes
back whenever the appendix is wanted.

**The sd is a third of the gap the specialist threshold sits in.** BBBP's
specialist sd is 0.0229 against a threshold 0.0353 below the mean, so a single
seed's specialist run could move that number by two thirds of the margin it is
meant to represent. Quote the three-seed mean, never one run, and carry the sd
wherever the specialist comparison appears.

**Only three of the nineteen tasks have a specialist**, and the rest needed a
convention. Settled: **the anchor is a trunk from another fold.** The folds
partition the tasks, so a task held out of fold A trains in the B, C and D
mixtures, and those runs' `in_mixture/<task>/test` scores are what a
generalist that *did* have the task in its mixture reaches — read off the
**annealed** model, which is the only reportable one, even though the legs fork
from the trunk checkpoint. The anchor is a level, not a starting point. That is the
comparison this study is about, it costs no extra run, and it applies to eleven
of the twelve owed tasks. The claim it supports is narrower than the specialist
one and gets stated as such: 95 % of a generalist's level, not of a specialist's.

**bond_path and longest_chain are the exception**, because they are held out of
*every* mixture rather than of one fold — no trunk ever trained them and no
specialist exists either. Their anchor is the from-base leg's own final score at
`budget_steps`. That is self-referential, and the ratio then reports "how much
sooner does the trunk reach what the backbone ends at" rather than "how much
sooner does it reach competence" — a defensible question and a different one.
Quote which one wherever the number appears. They pay for it with the one thing
no other task in the study gets: four independent parents. Since no trunk
trained them, legs off A, B, C and D are four readings of the same quantity
against one shared base leg, and the spread across folds is a free variance
estimate.

**Tox21 and SIDER have no anchor by design** (§1: they are gradient tasks, never
anchor numbers) and are not runnable as legs. With ClinTox, which has no train
split at all, that is three of nineteen out, sixteen legs in.

**A config whose threshold is not known yet carries `anchor`, not a number.**
`target` takes `value` or `anchor` and refuses both together; an `anchor` is a
string naming where the number will come from, and the leg then runs its full
budget, records every evaluation and reports a null crossing. The curves are the
durable artefact and the crossing is a reading off them, taken once the anchor
lands. This replaces the old `value: 0.0` placeholder, which was the worse
failure: 0.0 on a `max` target is met at the first evaluation, so both legs cross
immediately and the fork reports a ratio of 1.0 that looks like a result.

**The threshold travels on its own flag.** `--task` rewrites the metric key's
task segment and nothing else, so a Tier-B leg switched to BBBP kept BACE's
0.7792 and a generation leg switched to ChEBI-20 kept g2s's `roundtrip_match`.
Both now have flags — `--target-value` and `--target-metric` — and switching the
task while the config carries a threshold fixed for a different one is refused
rather than run.

## 7. Seeds

**Three trunk seeds on fold A first; one on B, C and D until the variance is
known.** One seed per fold cannot separate "pretraining helps" from "this trunk
was a good draw", the tasks inside a fold are not independent replicates because
they share a trunk, and a two-seed design cannot produce the seed-paired t on
2 df that every other result in this project reports.

What we already know about trunk-seed variance, from the three existing trunk
seeds scored on the permanent held-outs: `bond_path` EM 0.068 / 0.078 / 0.080
(sd 0.006), `clintox` average precision 0.373 / 0.423 / 0.366 (sd 0.031),
`clintox` accuracy 0.492 / 0.476 / 0.382 (sd 0.059). Structural tasks are tight
across trunk draws and property tasks are not. That is variance in the *starting
score*, not in the *speedup*, which is a ratio of crossing steps and can be
noisier than either input — measuring it is one of the things fold A is for.

**Put the cheap seeds in the adapt legs.** Those are 1–2 GPU-h each and the
crossing is a first-passage time, so fine-tuning noise lands directly on the
headline number.

**All three fold A seeds get legs, and that is the whole point of running
three.** Settled 2026-09-22. The first submission forked seed 0 only, which
leaves the three trunks measuring the variance of a *trunk score* and saying
nothing about the variance of the *ratio* — the quantity this study reports, and
the thing §7 says fold A exists to pin down. The other two seeds' legs went out
the same day. Fold A's four Tier-A tasks therefore carry three readings each,
and that is the only error bar anywhere in the study; B, C and D remain one seed
until these three say whether more are needed. The two permanent held-outs are
excluded (§6a), so this is four tasks per seed, not six.

## 8. What has to change in the code

Items 1–4 and 7 are **done**; 5, 6 and 8 are not, and none of them block a run.

1. **Fold-conditional holdout.** *Done.* `TRANSFER_FOLDS` in `config.py` is the
   fold table, and `molecule_generalist_mixture(exclude=…)` builds a fold's
   mixture by filtering the base one — renormalising Tier-B's `size ** 0.5` and
   recomputing `per_family` — so with an empty `exclude` it returns exactly what
   it always returned. The four presets are registered as
   `molecule_generalist_fold_{A,B,C,D}`.

   "Held out" is now a property of the parent, not of the registry, and
   `_plan_adapt` says so: a fork names `held_out_by: <fold>` and the task is
   checked against that fold's membership. Pointing a fold A leg at a fold B
   trunk is refused there rather than quietly measuring how recently the task was
   sampled. `is_held_out` still governs mixtures and is untouched.
2. **Every task gets train/val/test.** *Done, and it took three gates, not one.*
   `splits_for` was the obvious one. Under it sat `_draws`, which hardcoded both
   the split (`"held_out"`) and the pool (`_held_out_pool`) for a held-out Tier-A
   family, so a train-split build still drew held-out-role molecules and
   `_check_roles` correctly rejected them. Under *that* sat `schema.validate`,
   which refused any split but `held_out` on a held-out spec.

   Only one direction of that last rule was ever an invariant and it is the one
   that survives: `held_out` on a task that is *not* held out is still an error.
   The converse is gone, because an `adapt` fork trains on a held-out task — that
   is the mode — and without the ordinary three splits it trains and evaluates on
   the same thousand rows. What keeps it honest is the partition: one molecule,
   one role, and `_check_roles` refuses a train-split example whose key is not
   train-role.
3. **`clintox` stays held-out-only.** Unchanged, and now deliberate rather than
   pending. `_partition_claims` gives its molecules the `held_out` role and
   `_draw_tier_b` drops a train draw that is not train-role, so there is nothing
   to carve a train split from without moving `partition_version` and with it
   every split in the campaign. The two Tier-A families needed no partition
   change at all, so they were separated from it: `splits_for` returns all four
   splits for `bond_path` and `longest_chain`, and `("held_out",)` for `clintox`.
   ClinTox is therefore measurable in fold C from a trunk that never saw it, but
   only zero-shot — not as an adapt leg. Say so in the write-up.
4. **`steps_to_target` gains a persistence rule and a test-split exemption.**
   *Done.* `target.consecutive` (default 1) is how many evaluations in a row must
   meet the threshold; the step returned is the *first* of the run, because the
   question is when the model arrived and the rule only decides whether it
   stayed. A gap in the metric breaks the run rather than being skipped, since on
   a log-spaced grid two crossings an hour apart are not persistence.

   `target.on_test` admits a metric naming the test split. This is not a hole in
   D7.4: `target` is read exactly once, by `steps_to_target`, after both legs have
   run their full `budget_steps`. It stops nothing, picks no checkpoint and
   changes no weights — the reported quantity *is* the crossing step and the
   threshold is a constant fixed before submission. Reading it on val instead is
   not the conservative choice but the wrong one, because the threshold is a
   fraction of a specialist's *test* score and the val/test gap changes sign by
   task (BBBP val ~0.97 against test 0.7056; BACE val ~0.73 against test 0.8202).
   That mismatch is why the LR screen came back null on both arms. What the
   exemption does cost is optimism — a first-passage time over a noisy signal is
   biased early — so `on_test` *requires* `consecutive >= 2`.
5. **Log-spaced eval grid** for adapt legs. Not done, not blocking: the full eval
   `history` is stored, so any grid finer than the one used can be subsampled
   offline and any threshold re-applied.
6. **A plan-time retirement check.** Not done in the harness. Covered for this
   study by two scripts, one per side: `temp.gtds/fold_check.py` resolves each
   fold and compares `passes_needed` against the cap on each entry, and
   `temp.gtds/leg_passes.py` does the same for each adapt leg's one-task mixture
   and also checks what is built on disk. Both caught real failures — see §11.
   The leg-side one is the case worth putting in the harness eventually: it is
   the one nobody thought to look at.
7. **The micro-batch reshape had to be sized in tokens.** *Done, and it was the
   night's real blocker.* `align_to_accumulation` reshapes a step to exactly
   `accumulation_steps` groups, and it merged the *smallest* groups by example
   count. The sampler, meanwhile, buckets by padded length and caps a bucket's
   batch at `micro_batch_tokens // bucket`, so a long-sequence bucket comes out
   as many batches of **one** — which by example count are the cheapest things in
   the step and so exactly what the merge reached for first. Collapsing them
   rebuilt the oversized micro-batch the sampler had gone to the trouble of
   splitting: a 4 × 6,144-token group against a 1,024-token budget.

   It killed two of three fold A seeds inside forty steps, one at 72 GiB on an
   80 GB card. Cost is now `len(group) × max tokens in group` — the padded
   rectangle the collator actually builds — and a merge picks the pair whose
   merged cost is lowest, which keeps long examples apart instead of stacking
   them. `num_tokens` rides on each item from `MixtureDataset` as a fourth
   `SIDE_KEYS` entry; absent it, the cost degrades to the example count and the
   function behaves as it used to.

   **This changes the training stream**, so every trunk in the study has to be
   built under it. The trunks started before the fix were cancelled and
   resubmitted rather than kept.
8. **A target may defer its threshold, and the threshold travels per task.**
   *Done.* `target` now takes `value` **or** `anchor` — a string naming where the
   number will come from — and refuses both together, so eleven owed tasks can be
   queued without a stand-in number (§6a). `steps_to_target` returns `None` on a
   deferred target and the history is recorded either way. Alongside it,
   `--target-value` and `--target-metric` carry the two halves of a target that
   `--task` cannot infer, and switching the task while the config holds a
   threshold fixed for a different one is now refused. Both mismatches this
   guards against were live: `--task mol/bbbp` inherited BACE's 0.7792, and
   `--task mol/chebi20` inherited g2s's `roundtrip_match`, a key ChEBI-20 never
   emits. `temp.gtds/fork_target_check.py` resolves all sixteen legs through the
   real launcher path and asserts the refusal.
9. **An adapt `passes` may be a bare number.** *Done.* An anneal continues a
   whole mixture, so its `passes` has to name the task it raises; an adapt leg
   trains exactly one, so a scalar is unambiguous and `_plan_adapt` expands it.
   Without this, one fork config cannot drive nine legs under `--task` — the
   task would have to appear in the `passes` key as well, and the two spellings
   would drift.
10. **`held_out_by` is checked against the fold table by bare name, and
   `HELD_OUT_EVERYWHERE` is exempt.** *Done.* The table holds `ring_membership`
   and a task is `mol/ring_membership`, so the membership test matched nothing
   and would have refused every leg in the study. Separately, `bond_path`,
   `longest_chain` and ClinTox are held out of every mixture rather than of one
   fold, so every trunk is a valid parent for them and a one-fold membership
   test would have refused three parents of the four each legitimately has.
11. **`run_cli.sh` passes a colon-bearing `GPUS` entry through verbatim**, the
   same fix §11 records for `chain.sh`. It prefixed every entry with `GPU_BRD:`,
   so `GPU_MEM:80GB` became `GPU_BRD:GPU_MEM:80GB` and sbatch refused it — all
   twenty-two adapt legs failed to submit at once, with "submission failed; no
   job id" and nothing else.
12. **`align_to_accumulation` still cannot lower a peak it inherits.** If the
   sampler hands a step fewer groups than `accumulation_steps`, the reshape must
   split, and if it hands more it must merge — merging necessarily raises some
   group's cost. The fix minimises the maximum; it does not bound it. A hard
   per-micro-batch token ceiling belongs in the sampler, and is not owed by this
   study.

## 9. Run order

The ordering constraint worth exploiting: **the from-base adapt legs depend on no
trunk at all.** One from-base leg per task is shared by every fold, so they can
be built and queued immediately and run alongside trunk training. They are
single-GPU and short, which is exactly what backfills into fragmented
availability.

0. Code changes of §8, configs, pass arithmetic, dry runs. No GPU. **Done** —
   §8 items 1–4 and 7, all four fold configs, `fold_check.py` on every fold, and
   the full suite green at 1,053 tests.
1. Data prep for `bond_path` and `longest_chain` on the ordinary three splits.
   **Done.** ClinTox deliberately not included (§8 item 3). Nothing else needed
   building *for the trunks*: the fold mixtures do not move
   `partition_version`, because `_partition_claims` reads `config.pool` and
   `config.tier_b_corpora` rather than the mixture, so one partition serves all
   four folds.
2. Trunks: fold A seeds 0/1/2, then B, C, D. **Submitted.** Ahead of the adapt
   legs rather than behind them, which inverts the original order for a reason —
   the trunks are the long pole at ~22 GPU-h each and everything else is short
   and backfills around them.
2a. **A second prep, sized against the legs rather than the trunks.** Easy to
   miss and it nearly was: a leg gives one task every token, so it asks a
   generator for three or four times what the trunk did. One CPU job per task,
   48 passes for the nine Tier-A families and g2s, 120 for `bond_path` and
   `longest_chain` — see §11 for the numbers and for the two distinctions the
   sizing rests on.
3. **From-base adapt legs, sixteen tasks.** No trunk dependency and no longer
   blocked on the anchors: §6a's `anchor` lets a leg run its full budget and
   record its curve while its threshold is still owed, and eleven of the twelve
   owed thresholds are readings off trunks that are already queued. Single-GPU
   and short, so these backfill around the trunks.
4. Anneals, one per trunk.
5. From-parent adapt legs, per fold, as each trunk lands. `bond_path` and
   `longest_chain` get four each — no trunk trained them, so every fold is a
   valid parent (§6a).
6. Score, with `tools/kfold_score.py`. It fixes the owed thresholds off the
   landed anneals, reads the crossing off each stored curve under the same
   persistence rule the fork applies, and prints the parent/base ratio and the
   area beside it. It reports what is missing instead of failing, so it is worth
   running while the study is still going. Then decide from fold A's
   between-seed spread whether B/C/D need more seeds.

The three tools are one per stage and each is idempotent, so a half-finished
campaign is picked up by re-running them rather than by reconstructing where it
got to:

| tool | stage |
|---|---|
| `tools/kfold_adapt_all.py` | submits all twenty-two legs, held behind their trunks |
| `tools/anneal_all.py --queue-behind-trunk` | one anneal per trunk, same mechanism |
| `tools/kfold_score.py` | fixes the thresholds and reads the crossings |

**The anchor is read off the annealed model, not the trunk.** A trunk stops
mid-stable-phase by construction and is not the model any comparison should use;
the legs fork from the trunk checkpoint because that is where the parent's
*weights* are, but the anchor is a level rather than a starting point.

The fork configs are four, one per task kind, and `--task` drives the rest.
`--target-value` and `--target-metric` carry the parts of the target `--task`
cannot infer; see §6a and §8 item 8.

| file | tasks | why it is its own file |
|---|---|---|
| `adapt_kfold_tier_a.jsonc` | the nine training Tier-A families | `in_mixture` already scores their test split |
| `adapt_kfold_held_out_family.jsonc` | `bond_path`, `longest_chain` | held out everywhere, so `held_out` claims them and has to be asked for `test` |
| `adapt_kfold_tier_b.jsonc` | BACE, BBBP, HIV | `yesno`, so AUROC; and a corpus needs `passes` raised. Tox21 and SIDER have no anchor by design and do not run |
| `adapt_kfold_generation.jsonc` | `chebi20`, `g2s` | generative, so slower evals, a longer budget and a per-task metric |

Sixteen tasks run as legs, not nineteen: Tox21 and SIDER are gradient tasks and
never anchor numbers (§1), and ClinTox has no train split to fork on (§3).

## 10. Cost

A trunk plus anneal is ~22 GPU-h per seed (the three-seed campaign was 65).

| item | count | GPU-h |
|---|---:|---:|
| Fold A trunks | 3 | ~66 |
| Folds B/C/D trunks | 3 | ~66 |
| From-base adapt legs | 16 | ~25 |
| From-parent adapt legs | 16 + 6 | ~35 |
| **Total, one seed outside fold A** | | **~190** |

The from-parent count is 22, not 16: `bond_path` and `longest_chain` are held
out of every mixture, so each gets a leg off all four trunks rather than one.

True leave-one-task-out would be nineteen trunks, ~420 GPU-h of trunk alone, so
the clustering buys most of the design for a third of the price.

## 11. Open risks

- The pass arithmetic (§4). This is the failure that killed three 4× anneals at
  93 % in the assistant campaign, and the fold water-fill makes it more likely,
  not less. **It fired, and was caught before submission.** Resolved per fold by
  `temp.gtds/fold_check.py`:

  | fold | `budget_scale` | cap raised | why |
  |---|---:|---|---|
  | A | 2.0 | — | nothing exceeds its cap |
  | B | 2.0 | — | nothing exceeds its cap |
  | C | **1.25** | `mol/chebi20=8` | both, and for different reasons |
  | D | 2.0 | `mol/bace=12,mol/bbbp=12` | BACE/BBBP need 10 against a cap of 6 |

  Fold C is the instructive one. Dropping all of Tier-B enlarges ChEBI's design
  share from 20 % to 33.3 %, and at `budget_scale` 2.0 `resolve` refuses the
  mixture — ChEBI resolves to 16.7 %, half its design share and under the 80 %
  block floor. The error offers two repairs and **only one of them works**:
  raising ChEBI's repeat cap cannot move its share by a rounding error, because
  ChEBI is the *reference task*. `resolve` sets the scaled budget to
  `chebi_available / chebi_share × budget_scale`, so doubling the cap doubles the
  budget and the ratio is exactly where it started. Measured across caps 6, 8 and
  13 at a fixed scale the resolved table is identical to the last digit — same
  shares, same per-task examples, same 583,199-example budget. **A repeat cap is
  not a share knob when the task in question sets the budget.**

  So the scale comes down. The share goes as `1/(3 × budget_scale)`: 2.0 gives
  16.7 %, 1.5 gives 22.2 %, 1.25 gives 26.67 % against a floor of 26.67 %. 1.25 is
  the largest scale fold C admits and therefore the one to take. `max_steps` pins
  the horizon at 11,140 regardless, so what changed is the share mix, not the
  length. The cap still had to be raised — to 8, for the *retirement* reason,
  which the table's `epochs` column (7.59) does not report and should not be read
  as.
- **The pass arithmetic again, on the anneal side, where the table above does not
  reach.** The per-fold caps resolved above are the *trunk's*. The anneal fork
  carries its own `passes`, and `anneal_molecule_generalist.jsonc` — one file
  written for a trunk that trains all six finite corpora — raises all six. Fold C
  withholds five of them and fold D withholds ChEBI-20, so for those two folds
  the block names tasks their mixture never trains and `_with_passes` refuses the
  fork: 55 s in, exit 1. Folds A and B withhold none of the six and annealed
  cleanly, which is exactly why it went unnoticed until two of four cool-downs
  were missing. The refusal is right and should stay — silently dropping a
  `passes` key aimed at an absent task makes a mistyped task name
  indistinguishable from a working config.

  Resolved with one fork config per affected fold,
  `forks/anneal_kfold_fold_{c,d}.jsonc`, each naming only the corpora that fold
  trains. **The values are not the shared file's, and assuming they were cost a
  second submission.** A fork's `passes` pays for the tail on top of what the
  trunk already spent, so it reads against that fold's trunk row:

  | fold | trunk `task_passes` | anneal | why |
  |---|---|---:|---|
  | C | `mol/chebi20=8` | 9 | the 8 is consumed, not headroom; the decay wants 0.83 of a pass at ChEBI's 26.7 % share here |
  | D | `mol/bace=12,mol/bbbp=12` | 12 | 12 against the trunk's 9.62 already leaves 2.38 spare; the decay wants ~0.96 |

  Fold D is the case that makes the rule. The shared file's 7 is *below* that
  fold's trunk cap, so copying it is a reduction — refused, or worse, both
  corpora retired at once. And fold C shows the share arithmetic does not
  transfer either: the shared file's margins are computed at ChEBI's 19.30 %,
  against 26.7 % here. Never copy a per-task pass number across folds; both the
  cap already spent and the share it is spent at differ.
- **A fork is two legs in one job, and `run_cli.sh` allows two hours.** Neither
  an anneal nor an adapt fork fits: a Tier-A fork is 2 × 2,000 steps at ~2.2 s/it
  plus eighty evaluations, a generation fork is 2 × 3,000 plus sixty generative
  evaluations of up to 500 samples, and an anneal's decay is short but ends with
  the full validator suite over the whole mixture. Both launchers now set `TIME`
  — 6/10/20 h by fork config, 8 h for an anneal — because a fork that walls at
  90 % has produced nothing while an over-long limit costs only queue position.
- **The same pass arithmetic on the leg side, which is the harder case and was
  nearly missed.** A trunk splits its tokens nineteen ways; a leg gives one task
  all of them, so its demand is several times the trunk's for the same source.
  `temp.gtds/leg_passes.py` resolves each leg's own one-task mixture at its own
  budget and reads `passes_needed` back. It found three caps too low — BACE 37
  and BBBP 52 against 32, g2s 42 against 16 — and a build shortfall in every
  generator task, because the prep that served the trunks was sized against the
  trunks:

  | task | leg demand | built before | built now | note |
  |---|---:|---:|---:|---|
  | nine Tier-A families | 25–31 | 24 | 48 | just short, across the board |
  | `bond_path` | 98 | 24 | 120 | short examples, so many per step |
  | `longest_chain` | 109 | 17 | 120 | the worst of them |
  | `g2s` | 42 | 32 | 48 | |
  | BACE / BBBP / HIV / ChEBI-20 | 37 / 52 / 2 / 7 | n/a | n/a | corpora: a pass is a repeat, not a build |

  Twelve CPU jobs, one per task, closed it: the nine Tier-A families and g2s in
  20–42 minutes each, `longest_chain` in 1:41 and `bond_path` in 2:05. Running
  them in parallel rather than as one serial prep is what made the gap
  recoverable inside the trunks' own runtime.

  Two distinctions this rests on. A **generator**'s pass is a distinct draw and
  has to exist on disk; a **corpus**'s is a repeat of rows already built, so its
  `passes` is a cap and never a build requirement. And a cap set on a generator
  is not inert — `_with_passes` writes it onto the mixture entry, so the `16`
  the generation config first carried would have retired g2s two thirds of the
  way through its leg exactly as it would a corpus. Caps are now 64 in both
  files, and the builds went out as one job per task.

  The check also cost an hour to a wrong lookup worth recording: built passes
  live under **`build_version`**, not `partition_version`. The two are separate
  hashes on purpose — the partition is the expensive half and only moves when a
  key's role moves — and the partition has its own cache at
  `data/partitions/<hash>.json`, so looking there for pass directories finds
  nothing and reports every task as unbuilt.
- Folds C and D are structurally different trunks (§3) and must not be pooled.
  Fold C's measured block shares at the scale it runs at are Tier-A 45.8 %,
  g2s 27.5 %, ChEBI 26.7 %, against a renormalised design of 41.7/25/33.3 — so
  ChEBI sits on the floor and the other two run over, which is the opposite of
  what "drop Tier-B" sounds like it should do. Fold D resolves at exactly its
  renormalised design (Tier-A 38.5 %, Tier-B 61.5 %). Folds A and B are within
  3 % of design on every block, which is what makes them the poolable pair.
- **Card memory, not brand, is the scheduling constraint.** `GPU_BRD:A100`
  matches ana's 80 GB cards and axa's 40 GB ones alike, and a fold A seed that
  landed on axa died at step 13 with 37.9 GiB allocated out of 39.5 GiB. The
  fold configs now constrain on `GPU_MEM:{80,180,288}GB`, and `chain.sh` passes
  any `gpus` entry containing a colon to `--constraint` verbatim. Note that
  `GPU_SKU` values are not usable — sbatch refuses them outright ("Invalid
  feature specification"); `scontrol show node <n>` lists what exists.
- Molecule-level leakage: the partition is one molecule, one role, across all
  sources, so a held-out task's molecules are still in the trunk through other
  tasks. For A and B that is arguably the point — learning a new *question* about
  familiar graphs. For C it is more awkward, since that trunk saw BACE molecules
  without the BACE label. State it; do not try to fix it, because fixing it means
  a different partition per fold and no comparability at all.
- The speedup's between-trunk variance is unmeasured until fold A's three seeded
  leg sets land (§7). Those four tasks × three seeds are the only error bar in
  the study, and what they say settles whether B, C and D need replication.

  **They landed, and `kfold_score.py` was reading one of the three.** The row
  loop took `sorted(cells)[0]` per fold, so fold A reported s0 and discarded s1
  and s2 without a word — the replicated fold printed a single number in the same
  shape as the unreplicated ones, which is the failure mode a spread column
  exists to prevent. It moved the headline: `ring_count` reads 450/1,150 as the
  median of three, against 500/1,450 from s0 alone. The tool now scores every
  cell, reports the median, and prints the range beside it.

  The ranges are what the seeds were for, and they cut both ways. `ring_count`
  separates cleanly — parent 350–500 against base 1,100–1,450, no overlap — so
  its 2.9× is a property of the trunk and not of one run. `ring_size` does not:
  150–200 against 200–250 overlap, which is the evidence that its 1.0 is a
  ceiling artefact rather than a measured equality.
- **A `consecutive` rule can null a task that did reach the level.** BBBP scores
  `None` on both legs, and the reason is not that neither got there: the parent
  leg exceeded the threshold in 1 of 41 evaluations and the base leg in 7, but
  neither held it for the three consecutive readings the rule requires — longest
  streaks 1 and 2. At 1,244 molecules scored on 500 samples, BBBP's ROC-AUC
  bounces further than the margin being measured. Read this as the measurement
  being out of resolution on this task, not as an absence of transfer. What is
  genuinely informative is the direction: the **base** leg peaked higher than the
  parent (0.7180 against 0.6979), so there is no head start here to detect.
- **A fork config names validators; it does not configure them.** All four adapt
  configs were written as `[{"name": "in_mixture", "cadence": "steps:50"}]`, on
  the assumption that the 500-step cadence in the run config had to be overridden
  or a 2,000-step leg would get four evaluations. It does not: `validation_hook`
  runs a *named* validator with `event="manual"` at the fork's own `eval_steps`,
  whatever cadence it carries elsewhere, so `["in_mixture"]` was always the whole
  requirement. The object form parsed, planned, compiled and trained, then died
  at the first periodic evaluation on `set(names)` with `TypeError: unhashable
  type: 'dict'` — twenty minutes into a ten-hour job, six legs at once, on
  2026-09-22. `_validator_names` now refuses a non-string entry at plan time with
  a message naming the fix.

  Three things made it expensive out of proportion to the bug. A fork config is
  read **at job start**, so a whole queue can be wrong and look healthy until the
  first leg gets a GPU. The crash is **after** the compile and the first fifty
  steps, so it costs the expensive part of the job and not the cheap part. And
  `squeue` showed the six as `R` with a plausible elapsed time well after `sacct`
  had them `FAILED` — trust `sacct` for state, `squeue` only for what is
  scheduled.

  The `splits` and `max_samples` those configs also carried were inert for the
  same reason, and one of them was load-bearing: `held_out` scores the
  `held_out` split only, so `adapt_kfold_held_out_family.jsonc`'s
  `splits: ["test"]` was what would have produced
  `held_out/mol/bond_path/test/em_accuracy` at all. It never applied.
  **Re-enabling the two permanent held-outs as legs needs that solved in the run
  config, not the fork config** — as submitted they would have measured nothing
  and reported a null crossing.
- **An `adapt` leg measured every task except its own.** `build_eval_sets` walks
  the *parent's* mixture, and an adapt leg's task is by definition the one task
  that mixture does not contain. So it built sources for the fifteen tasks the
  leg does not care about and none for the one it does — and `in_mixture` takes
  its task list from `ctx.eval_sets`, so both halves followed:

  *The target was unreachable.* `in_mixture/mol/<task>/test/<metric>` was never
  produced, so every crossing would have come back `None`. A null there reads as
  "the trunk never got there", not as "this was never measured" — the study
  would have returned a clean, complete table of nulls. This was live from the
  first submission and hid behind two other failures.

  *And it was slow.* Fifteen tasks on two splits at 500 samples, the two
  generative ones included: **~13 minutes per evaluation** against 76 seconds
  per fifty training steps, which is ~19 hours over two legs against a 10-hour
  wall clock. Narrowed to the one task, a firing is **43 seconds** (measured
  twice on the probe leg, 50→51 and 100→101), and a job is ~2.6 h.

  `mode_fork` now calls `build_eval_sets(..., mixture=None,
  extra_tasks=(args.task,))` for `adapt` only — an anneal exists to produce the
  full suite — and refuses an empty result rather than training a leg with
  nothing to measure. That refusal is what surfaced this: it fired on the first
  job of the resubmitted batch, two minutes in, before any training.

  Two lessons, and the second is the expensive one. A wall clock was sized from
  the step count and the evaluation *count*, with the cost of one evaluation
  assumed rather than timed — **time one firing before sizing a fork**. And an
  absent metric and a missed target produce the same `None`, so **a null
  crossing has to be distinguishable from an unmeasured one**: check the metric
  key exists in a leg's first `history.jsonl` line before trusting any null in
  the table.
- **A leg's curve is written as it is taken, not at the end.** `result.json`
  lands once, after every leg has run, so a fork killed in its second leg used to
  lose the first leg's finished curve entirely — and `continue_leg` cannot
  rebuild it, since resuming a leg that already reached `max_steps` trains
  nothing and so evaluates nothing. `_run_leg` now appends each evaluation to
  `<leg>/history.jsonl` (rank 0, append-only), and `kfold_score.py` reads those
  back when `result.json` is absent, tolerating a half-written final line.
