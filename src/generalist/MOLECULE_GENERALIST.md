# The molecule generalist — one model over every molecule task

**Status (2026-09-02):** planned, not built. Gated on two things in the molecules campaign:
the tuned three-seed re-run that replaces every Tier-B number taken at `lr 3e-5`
(`src/experiments/molecules/PLAN.md` §8.4.4–8.4.5), and the harness in `DESIGN.md`. Nothing here
runs until both exist. This document is the *what and why*; `DESIGN.md` is the *how*.

**Where it sits.** This is arm 2 of the molecules plan (§4, "chemistry generalist") and the first
consumer of the generalist harness. Arm 1 (one specialist per task) is what the molecules campaign
has been measuring. Arm 3 (molecules folded into the cross-domain mixture) is the trunk's admission
gate and is *not* run separately — it would duplicate the trunk at the trunk's price.

---

## Summary

One 1B model, both arms (graph `rich_levi` and the SMILES flat twin), trained on every molecule
task the repo can produce — RDKit structural questions, MoleculeNet property prediction, ChEBI-20
captioning and graph-to-SMILES generation — with one molecule-level partition across all sources
and three held-out tasks that no training source ever touches.

**The result it produces:** arm 2 minus arm 1 on BACE / BBBP / HIV, i.e. whether free structural
labels improve scaffold-split property prediction. That claim survives any leaderboard outcome,
and no corpus-bound baseline can copy it. Secondary: zero-shot and adaptation-efficiency on the
held-out tasks, and the permutation-invariance spread of the flat twin.

**Why now, before the trunk:** the trunk is a multi-task model routed by the question node over
merged sources with per-source loss accounting. None of that has been exercised on molecules. One
domain with two metric families (exact match beside AUROC-from-margin), a multi-endpoint set, a
captioning target and a string-generation target is a harder test of the schema and the mixture
code than any single graph-QA domain — and it costs a few percent of what the trunk costs.

### Checklist

- [x] **Gate 1.** ✅ **2026-09-03**, `molecules/PLAN.md` §8.4.8: all three datasets, 18/18. Arm 2 is
      compared against *these*, not against the `lr 3e-5` tables.
- [x] **Gate 2.** ✅ D1–D6 built and the plumbing smoke run passed (`results/BUILD_LOG.md` §T10).
- [x] **Build.** ✅ Molecules adapter, the partition, graph-to-SMILES generator, ChEBI-20 loader,
      the isomeric round-trip test.
- [x] **Cross-check.** ✅ **2026-09-03** (`002` / `003` + `forks/anneal_cross_check.jsonc`). BACE
      seed 0 as a single-task mixture, both arms, 40 passes then a 151-step anneal. End-of-anneal
      test ROC-AUC: **graph 0.8034, flat 0.8264**, against §8.4.8's three-seed specialist rows
      (graph mean 0.8202 sd 0.0120; flat mean 0.8224 sd 0.0248). Flat lands +0.16 sd from its mean;
      graph lands 1.4 sd low, just outside the three-seed range and inside seed noise — the thinner
      of the two margins, recorded as such. **The arm difference, which is what arm 2 reports,
      reproduces to 0.002**: harness −0.0230 against the specialist's seed-0 paired −0.0250. Splits
      verified molecule for molecule, budgets and examples-per-step matched to `026`. Full write-up,
      the five defects it found and the Property-1 measurement in `results/BUILD_LOG.md` §T11.
- [ ] **Before arm 2.** A multi-GPU shakedown — ~100 steps of `001` at the target GPU count, since
      nothing has run distributed and every defect so far came from walking a path the first time.
      Settle the `in_mixture` firing cost, which the single-corpus cross-check never exercised. A
      general-text held-out loss, adapter-on against adapter-off: the assistant goal makes text
      ability something to measure, and no validator measures it. Optionally two more cross-check
      graph seeds, ~25 min each, for a margin that landed thin.
- [ ] **Arm 2.** Three seeds × two arms, the mixture in §2, WSD stable phase, one anneal fork at
      the end.
- [ ] **Read-out.** Arm 2 − arm 1 per (dataset, seed), paired; held-out zero-shot; adaptation
      steps-to-target from the generalist vs from base Llama; permutation spread.
- [ ] **Write-up** into this file, with the disclosures §7 lists attached to every number.
- [ ] *(Optional, §8)* the same recipe on Llama-3.2-3B and Llama-3.1-8B.

---

## 1. What goes in

Both arms train on everything below. The graph arm is `rich_levi`, `stereo_tags: on`, no SMILES
anywhere in the prompt — that is what makes graph-to-SMILES a real task rather than a copy. The
flat twin sees SMILES and gets the matched form of each task. Every source is routed by the
question text alone (the question node is `on`, D3), so the model receives the task only through
the question.

| Source | Tier | Task form | Size | Role | Metric |
|---|---|---|---|---|---|
| `ring_membership`, `aromatic_ring`, `ring_size`, `ring_count` | A | 1–3 token exact answer | generator, capped per pass | train + in-mixture test | exact match |
| `fg_presence`, `fg_count`, `fg_atom_membership` | A | 1–3 token exact answer | generator, capped | train + in-mixture test | exact match |
| `stereo_potential`, `stereo_assigned` | A | 1–3 token exact answer | generator, capped | train + in-mixture test | exact match |
| BACE, BBBP, HIV | B | yes / no | 1.5k / 2.0k / 41k molecules | train + **headline** test | ROC-AUC from the yes/no margin |
| Tox21, SIDER | B | yes / no per endpoint | 78k / 39k (molecule, endpoint) pairs | train + diagnostic test | ROC-AUC per endpoint |
| ChEBI-20 | C | free-text caption | 26.4k / 3.3k / 3.3k | train + in-mixture test | BLEU-2/4, ROUGE-L, METEOR |
| graph-to-SMILES | — | canonical SMILES, stereo-free (§5) | generator, every train-role molecule once per pass | train + in-mixture test | validity, round-trip match, canonical exact match |

**Tox21 and SIDER are training signal, not results.** They were deferred from Tier B because they
have no anchor ladder (`molecules/PLAN.md` §1), which is a problem for *interpreting* a number
against the field and no problem at all for gradient. Together they are the bulk of the available
property labels, they take arm 2 from three endpoints to about forty, and Tox21 is the only thing
in scope that exercises multi-endpoint routing. Their test AUROC is reported as an internal
arm 2 vs arm 1 diagnostic and never placed in the anchor table. Tox21's ~16k absent labels are
skipped at the (molecule, endpoint) level; the per-endpoint example counts go in the run record,
because skipping changes each endpoint's effective weight silently otherwise.

**`stereo_assigned` goes in on a non-degenerate pool only** and keeps its job as the suite's
leakage detector: at chance with the parity channel closed, high with it open. On the `014` pool it is
single-answer (§3.2.10.1) and would be void; the generalist's pool is the train-role union (§3),
which had to be measured for this family before the mixture was frozen.

Measured on build `42f7a14bed21f876`: the pool is 44,088 train-role and 5,503 test-role molecules, and
the family is **not** degenerate here — ten distinct answers on both roles, against `014`'s one. It is
still heavily skewed. The drawn test split (1,000 molecules) answers 0 on 927 of them, so the floor is
0.927 and the whole headroom is 7.3 points; train draws sit at 0.948. Two consequences for reading the
detector. At the validator's 500 scored rows the 3σ line lands at 0.962, three and a half points above
the floor, so a verdict costs half of what headroom there is. And only the ~7 % of rows with a nonzero
answer change at all when the parity channel is closed — about 36 rows out of 500 carry the entire
signal, and `n_stripped` is reported alongside the gap for exactly that reason. The family stays in at
its 0.0278 share; what it cannot support is a *fine* reading, so the verdict is a floor test and
nothing more.

The detector itself is now built, as the `leakage` validator (`evaluate/builtin.py`), and it is in the
default set. It closes the channel at *evaluation* rather than training a second model the way
`molecules/configs/016` would have: the parity words come out of the graph arm's node text, and the
flat arm's SMILES is re-serialised without stereochemistry, since `stereo_tags` was never a flat-arm
knob. Two scoring passes instead of a training run, and it catches the failure the control exists for
— a memorised molecule is answered from memory whether or not its parity words are present, which is
how §3.2.10's duplicate test items showed themselves. What it does not reproduce is `016`'s off/on
*training* contrast, and the difference is in the safe direction: a model trained with the tags on can
also lose accuracy simply because its input moved, which can only push the stripped score down toward
the floor. The floor is the split's own majority-class rate, measured at score time rather than
inherited from `016`'s 0.732, because this pool is a different one; a single-answer split reports
`void` rather than a pass, which is the `014` state and the one reading that must never look like
a clean bill of health.

**Excluded.** ESOL, FreeSolv and Lipophilicity as *tasks*: the margin readout cannot score a
number and none of them has an anchor. Their molecules still serve as unlabeled pool for the
generators. QM9, peptides and text-to-molecule stay out for the reasons in `molecules/PLAN.md`
§1. The `+smiles` graph arm is out: it would turn graph-to-SMILES into a copy task and the
permutation-invariance claim into a lie.

## 2. Mixture

Weights are in *examples*, and by D7a each task's gradient share equals its example share
(two-level normalization, per-example within a task). Starting point, to be recorded in the
registry and revised only against the per-source loss curves the smoke run produces:

| Block | share | within the block |
|---|---:|---|
| Tier B (5 sets) | 0.40 | temperature ∝ size^0.5 over the five sets — roughly BACE 5 %, BBBP 6 %, HIV 27 %, Tox21 37 %, SIDER 26 % of the block |
| Tier A (9 families) | 0.25 | uniform over families |
| Tier C (ChEBI-20) | 0.20 | — |
| graph-to-SMILES | 0.15 | — |

**Passes.** Finite sources (Tier B, Tier C) get at most **six** passes. Generators (Tier A,
graph-to-SMILES) draw fresh examples every pass from the train-role pool, single-pass, so the
early-peak overfitting the specialist runs show cannot come from repetition. The total budget is
therefore *defined by the finite sources*: six passes of Tier B + C at their combined 0.60 share
fixes the number of examples, and the registry computes and records the step count from that
rather than taking it as a free knob.

Six, and not the three this section originally specified, because of how the budget rule interacts
with the temperature weighting. The budget is `min over finite corpora of (passes × train_size) /
share`, and within Tier B the weight goes as `size ** 0.5` while the cap goes as `size` — so
`available / share` scales as `size ** 0.5` and **the smallest corpus always sets the horizon**. At
three passes that was BBBP: 1,244 training molecules and 2.35 % of the run ended training at 2,799
steps, with every large corpus far from its own cap.

| task | share | epochs at 3 | epochs at 6 |
|---|---:|---:|---:|
| `mol/bbbp` | 0.0235 | **3.00** (binds) | **6.00** (binds) |
| `mol/bace` | 0.0203 | 2.67 | 5.35 |
| `mol/chebi20` | 0.2000 | 1.55 | 3.11 |
| `mol/sider` | 0.1036 | 1.07 | 2.14 |
| `mol/hiv` | 0.1062 | 0.52 | 1.04 |
| `mol/tox21` | 0.1465 | 0.43 | 0.86 |

HIV at 0.52 epochs was the sharp end of it: the specialist HIV cell that arm 2 is differenced
against trained roughly ten, so a generalist deficit on HIV would have been partly a budget
artifact. Six doubles the horizon to **5,599 steps** and takes HIV just past one epoch. Raising
BBBP alone would not have worked — BACE simply inherits the binding role at 3.00 epochs and the
budget moves 12 % — so the cap moves for the finite corpora as a set.

This is a ceiling, not the fix. A small corpus should be *drawn less often*, not allowed to end the
run when it is exhausted; the correction belongs on the sampling side, in the weight rather than in
the cap. That changes the mixture shares every number so far was measured under, so it waits for
the campaign after this one rather than landing between arm 1 and arm 2. It is worth doing before
the larger generalists, where the same rule would bind harder.

**Loss.** Per-example normalization everywhere (D7a default). Captions are 50–100 tokens beside
one-token answers; token-summed loss would make Tier C most of the gradient at a fifth of the
examples. No per-task escape hatch is used here; the field exists in the registry for CLRS and is
left at its default.

## 3. The partition — one molecule, one role

The Tier-A generators and graph-to-SMILES draw molecules from the Tier-B corpora. Without a single
rule across sources, a structural question about a BBBP *test* molecule lands in training and the
scaffold split stops meaning "structurally novel". The campaign already had exactly this incident
once (`molecules/PLAN.md` §3.2.10). So:

* **Key:** *stereo-free* canonical SMILES from RDKit, computed once per molecule at adapter build
  time. Stereo-free on purpose: two stereoisomers have identical graphs up to the parity words, so
  keying on the isomeric string would let near-identical graphs straddle the train/test line.
  Both isomers therefore share one role; each keeps its own labels.
* **Roles:** `train`, `val`, `test`, `held_out`. Every molecule in every source gets exactly one.
* **Rule 1.** A molecule in any Tier-B val/test split, any ChEBI-20 val/test split, or anywhere in
  ClinTox is removed from *every* training source — Tier-B train splits of other sets, the Tier-A
  generator pool, graph-to-SMILES, ChEBI train. Priority on conflict: `held_out` > `test` > `val` >
  `train`.
* **Rule 2.** Generators draw training molecules only from the `train` role.
* **Rule 3.** Generator *test* sets draw from `test`-role molecules, which are scaffold-novel by
  construction. This is what the Tier-A re-run (`014`) already does.
* **Rule 4.** The registry refuses to build a mixture whose sources violate rules 1–3, and the run
  record carries the per-role molecule counts and the number of cross-source overlaps removed.

Enforced in `src/generalist/adapters/molecules.py` and pinned by a test that builds the partition
from the raw CSVs and asserts pairwise disjointness of the role sets (`DESIGN.md` §T2).

**The `val` role is larger than it needs to be — shrink it next rebuild.** Rule 1 removes every
val-role molecule from *every* training source, so the 7,690 molecules currently in that role cost
training data across the whole mixture, not just in the set they came from. They buy a diagnostic
that never selects anything: WSD has no dev-score checkpoint selection, the reportable model is the
end of the anneal, and §8.4 measured Tier-B validation *anti-ranking* the arms on BBBP. A few
hundred rows per source would read the same curves at a fraction of the cost, and the difference
returns to `train` — the same currency the pass-cap problem in §2 is denominated in. Left as it is
for this campaign on purpose: the partition is an input to `build_version`, so changing it means a
full rebuild and a new build id, and it is not worth invalidating six cells mid-flight for. Fold it
into the next rebuild, alongside §2's move of the pass cap into the sampling weight.

## 4. Held out

| Held out | Why this one | Scored as |
|---|---|---|
| `bond_path` | Declared 2026-08-28 (`molecules/PLAN.md` §4.1). SPD *is* the answer by construction, and SPD is the graph arm's bias, so it is the cleanest test of whether the structural channel crosses question templates. | zero-shot exact match, then steps-to-target from the generalist vs from base Llama |
| `longest_chain` | Added 2026-09-02. With `bond_path` it makes the held-out set *the traversal family* while training covers rings, functional groups and stereo. Transfer from local motifs to path questions is a real claim; `ring_count` was the alternative and is weaker because `ring_size` and `ring_membership` are in training, so transfer there is near-duplicate. It remains measured as a specialist (`014`: graph 0.988 vs flat 0.828), so nothing is lost by holding it out. | same two ways |
| ClinTox | Declared 2026-08-28. A toxicity / trial-failure endpoint, unlike binding or permeability. | zero-shot AUROC, then steps-to-target |

Two Tier-A holdouts is the number. A third starts costing training coverage for a declaration
made after seeing results, which is worth less than the two made before.

The adaptation runs are three held-out tasks × two starting points (the generalist and base
Llama) × **three seeds** — eighteen short runs, and three is the seed count every other claim in
this campaign is quoted at. Steps-to-target is a first-crossing statistic and noisier than an
end-of-run score, so a single seed would not separate a real gap from where the curve happened to
cross.

Few-shot means *few-example fine-tuning* (the adaptation curve), not in-context examples. Several
molecule graphs in one prompt is not something the prefix-node layout is built for, and the
in-context anchor in the Tier-B table (Vicuna-13B, 4-shot) sits at chance anyway.

The molecules package already refuses to build `bond_path` and ClinTox without `held_out_eval`
(`HELD_OUT_TIER_A_TASKS`, `HELD_OUT_DATASETS` in `data.py`); `longest_chain` joins those tuples,
and the generalist registry mirrors all three so a mixture that names any of them fails in both
places.

## 5. Graph-to-SMILES

The one task not in the molecules plan. The graph is on the *input* side, so it is the inverse of
the text-to-molecule generation `molecules/PLAN.md` §1 excludes, and it is the bridge to
captioning: a model that can write a molecule's structure is better placed to describe it.

* **It is not one-to-one, so the target is RDKit canonical SMILES**, which makes it a function of
  the molecule. Three metrics, in order of what matters: validity (RDKit parses the output),
  round-trip match (parse, canonicalize, compare to the target), canonical exact match (the strict
  proxy; canonical atom ordering is an RDKit ranking the model may or may not learn to reproduce).
* **Stereo is out of the target, and the reason is structural, not a shortcut.** The node text
  carries the tetrahedral parity word (`cw` / `ccw`) and the bond stereo word (`E` / `Z`), but a
  parity word is only meaningful relative to a neighbour *ordering*, and the graph has none — that
  is what permutation invariance means. `roundtrip_check` says so in as many words and compares
  stereo-flattened strings for exactly this reason (`data.py`, the `exact` level). A graph arm
  asked for `@`/`@@` would be asked for information it does not have. So the target for **both
  arms** is the stereo-free canonical SMILES, `Chem.MolToSmiles(mol, isomericSmiles=False)`,
  which keeps the comparison matched: the flat twin's input carries stereo it must learn to
  *drop*, the graph arm's input carries parity words it must learn to *ignore*. E/Z is in
  principle recoverable from connectivity through CIP priorities, so a later cut may add it back
  once the round-trip test reconstructs it; tetrahedral chirality would need an order-independent
  parity encoding (parity relative to canonical atom rank), which is an encoding decision for
  the molecules plan, not this document. Whether the model emits stereo marks at all is recorded
  as a diagnostic, since emitting them is an error under this target.
* **The flat twin's matched task is canonicalization:** randomized SMILES in, canonical SMILES
  out. The graph arm has no input order to randomize, so it faces the hard version by
  construction — the atom-order invariance property showing up as a task.
* **It is a generator with free labels** and is capped at one example per train-role molecule per
  pass, at the §2 share, or it swamps the mixture.
* Question text: `Question: write the canonical SMILES for this molecule.` No atom labels.

## 6. ChEBI-20

* Keeps its own split (26,407 / 3,301 / 3,300) and is folded into the §3 partition. Overlap with
  MoleculeNet is expected to be small and is *measured*, not assumed.
* Per-example loss, as §2.
* ChEBI-20 includes salts and multi-fragment molecules. Disconnected graphs put SPD at the
  `max_spd` clamp between components, and larger molecules hit the node budget. Both are checked
  at build time; a heavy-atom cap is chosen against the ChEBI size distribution and recorded.
* The templated-caption caveat from `molecules/PLAN.md` §1 stays attached to every Tier-C number:
  BLEU and ROUGE reward template matching, so a strong number is weak evidence.

## 7. Recipe, measurement, and what gets reported

**Recipe.** Both arms at `lora_r 16` (the r32 axis is closed, §8.4.5), `lora_dropout 0.05`
(the molecules value, so arm 1 and arm 2 match; the trunk's D3 value of 0.15 is not used here),
`bias_lr 1e-2` on the graph arm, `weight_decay 0.1`, `max_spd 32` — settled, and settled at the value
this document already carried (`molecules/PLAN.md` §8.4.6, ablation §8.4.9).

**The learning rate is `1e-4`, matched across arms — settled 2026-09-04.** The specialist settled it
per (task, arm), 3e-4 on BACE and BBBP against 1e-4 on HIV (§8.4.6), and one mixture cannot hold
both, so the only question is which of the two a single rate inherits. It inherits the lower one for
two reasons. **The schedule is not the one those rates were tuned on:** the specialists ran warmup +
cosine, which touches its peak for a moment and spends most of the run below it, while WSD holds the
stable phase at `lr` for essentially the whole run. The same number is a materially larger dose here,
so carrying a cosine peak across as a constant rate is not a matched transfer — it is the §3.2.7 and
§8.4.4 mistake in a third place, an inherited number used across a change in the thing that gives it
meaning. **And the risk is asymmetric:** 3e-4 was measured to cost the graph arm 0.109 ROC-AUC on
HIV (§8.4.7), where it also produced the screen's highest validation score and its lowest test score,
while 1e-4 on BACE and BBBP was screened and came out worse rather than broken. HIV is 10.6 % of this
mixture against BACE's 2.0 % and BBBP's 2.4 %. `lr_min` follows it to `1e-5`, keeping the anneal at
`lr/10`.
Schedule: WSD — short warmup, constant stable phase for the §2 budget, one anneal fork at the end
that decays to `lr/10` over ~10 % of the stable steps. The annealed checkpoint is the reportable
model. There is **no test-set selection and no best-val selection on the generalist**: Tier-B val
anti-ranks arms on BBBP (§8.4) and the anneal fork makes selection unnecessary.

**The comparison.** Arm 1 records both its best-val and its last-checkpoint test score
(`test_roc_auc` and `test_roc_auc_last`). Arm 2 minus arm 1 is reported against *both*, paired
within (dataset, seed), with the last-checkpoint pairing as the primary because it is the one
free of a selection instrument shown to be near-blind.

**Two claims, never conflated** (`molecules/PLAN.md` §8.4.3.1): mean ± s.e. over seeds on the
fixed test set, which is what the anchors publish; and the paired per-molecule bootstrap, which
is the generalisation claim. HIV's effective n is ~132 actives, not 4112 molecules.

**Disclosures that travel with every number:** Tox21 / SIDER are not anchor-comparable; the
Tier-C caveat; the flat twin's graph-to-SMILES is canonicalization; the partition counts.

**Free extras.** Permutation-invariance spread of the flat twin on 10 randomized SMILES per test
molecule, stratified by symmetry class (`molecules/PLAN.md` §6) — one eval pass. Adapters-off
bit-exactness against base Llama (Property 2) at every milestone.

**Cost.** Set by the §2 budget rule and measured, not estimated, by the smoke run. The HIV
specialist at 10 epochs (~10k steps) is the anchor for one seed of arm 2's order of magnitude;
three seeds × two arms is the whole of the compute.

**The six cells, and where they live.** `configs/runs/001_molecule_generalist_{graph,flat}_s{0,1,2}.jsonc`,
launched one file at a time through `tools/chain.sh`. The graph seed-0 file carries the reasoning for
the whole campaign and the other five state only their delta, so a recipe decision has one place to be
corrected; `test_the_campaign_cells_differ_only_where_they_are_meant_to` asserts the six agree on every
field but the run name, the seed, and the three that separate the arms. Both arms read one build,
`42f7a14bed21f876` — `build_version` is a function of the data fields alone, and all six hold those
fixed, so a difference between two cells is the run's own spread and nothing else.

**Resolved budget** (build `42f7a14bed21f876`): **5,599 steps** on both arms — 318,217 examples on
the graph arm, 318,179 on the flat one. Bound by `mol/bbbp`, the smallest corpus, at its six-pass
cap; see §2 for why the cap is six and why the smallest corpus is always the one that binds.
`max_steps` pins the horizon explicitly on all six cells rather than leaving it to the budget rule,
for the reason in the next paragraph.

**The arms are matched in examples, not tokens, and that costs a second number.** D4.4 sets the batch
in tokens so a mixture of very differently-sized tasks costs a roughly constant amount per step. That
is the right default and exactly wrong for an arm comparison: the built graph mixture measures 288.28
tokens an example against the flat arm's 82.51, so one shared token budget would hand the flat arm 3.5×
the batch. The graph arm runs at `tokens_per_step 16384` and the flat arm at **4689**, which is the
value that lands on the same ~56.83 examples/step (`tools/tokens_per_step.py`).

At this budget no integer token count lands the flat arm on the graph arm's step count as well:
4689 resolves to 5,600 steps and 4690 to 5,598, straddling 5,599 without touching it. So `max_steps`
pins 5,599 for both arms and the flat arm takes the value on the *short* side — 4689 draws 318,179
examples where 4690 would ask for more than the budget holds. The pair therefore matches exactly on
schedule length and differs by 0.012 % on examples per step, which is the right way round: a step
count is what the WSD phases are measured in.

**One harness defect the shakedown caught, and it is not a small one.** `torch._dynamo`'s recompile
cap was left at the model default of 32, which training never approaches — one bucket ladder, a
fixed micro-batch, a handful of `(L, N)` pairs. Evaluation is a different regime: a milestone firing
sweeps sixteen tasks across two splits whose length profiles have nothing in common, so it walks
through more distinct shapes than the whole of training. Past the cap dynamo does not raise — it
drops `flex_attention` to the unfused eager path, which materializes the full scores matrix. The
2026-09-05 probe hit it at step 100 and spent over an hour in a validator block against 1.55 s/step
of training. The cap is now 128 (`wiring.FLEX_CACHE_SIZE_LIMIT`), and eval batches keep their row
counts on a power-of-two ladder so the batch dimension stops contributing shape variety of its own
(`evaluate/scorers.py`). Worth remembering as a class of bug: a compile cap that is too low costs an
order of magnitude and reports nothing but a warning.

## 8. Optional last step — a larger suite

After the 1B result is in, and only then: the same registry, mixture, partition and harness on
**Llama-3.2-3B** and **Llama-3.1-8B**. Same adapter (`modeling_gtlm_llama.py`), same tokenizer
family, so nothing but the backbone moves. One config each, both arms, one seed first. The
Gemma-3 adapter exists but changes tokenizer and attention layout at the same time, which is a
different experiment.

This is not a result the plan needs. It is worth doing because a small suite of graph-aware
molecule models at three sizes is useful to other people, and because the 1B-vs-larger delta on
the *held-out* tasks is the one place scale could sharpen a transfer claim rather than just a
SMILES-reading one. It does not run until the 1B write-up exists, and it inherits every gate above.
