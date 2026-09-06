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

**What it produced, 2026-09-06 (§8).** The primary claim is a **null at the available resolution**:
pooled over nine (set, seed) pairs against the last-checkpoint pairing, the generalist beats the
specialist by +0.0100 ROC-AUC on the graph arm and +0.0154 on the flat one, against a per-pair sd of
0.025–0.040 and a power table that puts three seeds at 0.05. Training on every molecule task at once
neither helped nor hurt property prediction measurably. The secondary claim is the strong one:
**Property 1 passes on all fifteen graph cells**, with the graph arm pinned at two to three grid
steps of the margin's quantum while the flat arm runs 10× to 40× wider — and the gap *widens* through
the anneal, because the flat arm's spread grows with its margins and the graph arm's does not. Within
arm 2 the arms split along one line: the flat arm wins property prediction (0.7969 vs 0.7795), the
graph arm wins the structural probes (0.9408 vs 0.9359) and both held-out structural tasks.

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
- [x] **Arm 2.** ✅ **2026-09-06.** Three seeds × two arms at 5,599 steps, the mixture in §2, WSD
      stable phase, one anneal fork a cell decaying to `lr/10` over 561 steps. All twelve runs
      completed; the six annealed checkpoints are the reportable models.
- [x] **Read-out.** ✅ §8. Arm 2 − arm 1 paired within (dataset, seed) against §8.4.8; held-out
      zero-shot; permutation spread on both arms. Adaptation steps-to-target is **not** measured —
      it needs an `adapt` fork per held-out task and none was run.
- [x] **Write-up** ✅ §8, with §5's graph-to-SMILES disclosure and §7's generation-path caveat
      attached to the numbers they bear on.
- [ ] *(Optional, §9)* the same recipe on Llama-3.2-3B and Llama-3.1-8B.

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

**A second one the campaign caught, and it was reporting numbers rather than hiding.** The recompile
cap explained a slow validator; it did not explain a validator that stays slow with the cap raised.
The first milestone of the campaign's own run cost **133 minutes on the flat arm and about 161 on the
graph arm**, against 34 minutes of training for the 2,000 steps it interrupted — and the second firing
cost the same as the first, so it was not compilation. It was the generation loop, which ran one
example per `generate` call. Pulling on that found the reason it had to: a generation batch was
believed unbatchable because "the prompt node must stay last in the packed sequence, so nothing may
be padded past it". True, and not a statement about batch size — `GraphCollatorV2` under
`pad_to_block` rounds the packed length up to a 512 bucket, so a 200-token prompt is followed by 312
pads *at batch size one*. The padding side was the constraint all along, and the padding side is a
knob.

Two things came out of moving it (`GraphCollatorV2(padding_side="left")`, `evaluate/scorers.py`):

* **The generative metrics were wrong, in the direction of understating the model.**
  `prepare_inputs_for_generation` numbers each new token `position_ids[:, -1] + 1`, continuing the
  prompt node's local counter. Under right padding `position_ids[:, -1]` is a pad, which is 0, so
  every continuation was numbered 1, 2, 3 … and RoPE placed it *before* the prompt it was answering.
  Measured on `checkpoint-1500` of `graph_s0` and `checkpoint-3500` of `flat_s0`, 128 samples, every
  teacher-forced number identical and only the generative ones moving: flat-arm ChEBI-20 METEOR
  0.285 -> 0.463 and ROUGE-L 0.213 -> 0.303; flat-arm g2s `roundtrip_match` **0.000 -> 0.031** and
  `exact_match` 0.000 -> 0.008. A metric that read as "the model cannot do this at all" was an
  artifact of where the continuation was placed.
* **It is between eight and eleven times faster**, on the same checkpoint, rows and card: 3,946 s ->
  464 s on the graph arm, 3,341 s -> 298 s on the flat arm (second pass in both cases; the first pays
  for compiles).

**Flex on the generation prefill is a memory result, not a speed one.** `use_flex` needs
`q_len == kv_len` and a block-aligned length, and a bucketed prompt batch is both — so forcing
`graph_attn_impl` to eager was giving up the fused kernel on the one quadratic pass in the
evaluation. Batched-and-eager against batched-and-flex came out level on the clock (458 s vs 464 s,
and the eager run was on the *faster* comparison in one sense — a B200 against a B300 — so flex is at
worst neutral), because generation is dominated by sequential decode steps and decode falls back to
eager inside the model either way. The peak is where it shows: **23.8 GB batched-eager against 10.5 GB
batched-flex**, with the old one-at-a-time path at 13.1 GB. So batching *without* flex would have
pushed the evaluation's peak above the path it replaced, next to a training step that already peaks
at 100.6 GB on a 178 GB card — which is the shape of the 2026-09-04 OOM. Flex on the prefill is what
makes the batching safe to run mid-training.

**Evaluation is sharded across ranks now** (`evaluate/parallel.py`). The unit is a whole scoring
target — a `(task, split)` pair, one stereo view, one task's permutation sweep — never a slice of
one, so a rank scores the same rows in the same batches it would have alone and the metric is
identical rather than merely close. Verified at four ranks against one on the same card class: **889
of 889 keys identical**, all four ranks agreeing with each other, at 918 s -> 268 s (3.4x of an ideal
4x, the gap being the imbalance left by scoring whole targets). Placement is
longest-processing-time-first against a cost estimate that prices a generated row at `max_new_tokens`
forward passes and a teacher-forced one at one.

Together: a milestone that cost 133 minutes on the flat arm should cost about **12 minutes** on one
card and **3-4 minutes** on four. The single-card campaign cells are unaffected — with one rank there
is no process group and no gather, and the code takes the direct path.

**Shape count, since the batching adds a family of them.** Every batch the evaluation collates was
recorded as its `(B, L, N)` triple, which is what the compiled flex kernel guards on. The whole
evaluation touches **25 distinct shapes on the graph arm and 1 on the flat arm**, against a cap of
128 — and that is *fewer* than the path it replaces (27 and 2), because generation now groups on the
same power-of-two row ladder and the same L/N buckets as the scoring batches instead of contributing
a `B = 1` family of its own. The cap does not need raising.

**Two rank races, found by running a fork distributed for the first time.** `prepare_fork` and
`_write_result` both run on *every* rank and both are filesystem writes. The copy of the parent
checkpoint was guarded by `if not os.path.exists(...)`, which is check-then-act: two of the six
anneal cells died with `FileExistsError` when a second rank created the directory between the check
and the `copytree`. The crash was the good case — a rank losing the race the other way skips the copy
while rank 0 is still making it and resumes from a half-copied checkpoint, and four ranks rewriting
`schedule.json` at once can tear the file that decides the whole decay. `result.json` showed the
silent version: it is an `open(path, "w")` every rank reaches, and four of the six cells wrote a file
that was one complete document overwritten by another at a different offset. One was recoverable
from its own tail, three were not and came back from the copy `mode_fork` prints to stdout. Both
sites are rank 0 only now, with the other ranks waiting on a marker rank 0 writes last; `append_line`
needed no such guard, being `O_APPEND` under `flock`, so concurrent ranks duplicate a lineage entry
rather than tear one. The rank has to come from `RANK` in the environment rather than
`evaluate.parallel.world`, because the process group does not exist yet at `prepare_fork` — it is
created inside the first `TrainingArguments`, which `_run_leg` builds afterwards, so `world()` would
answer 0 on every rank and every one of them would write.

**DDP width is not free on the graph arm, and the reason is the shape set.** The anneals ran at four
ranks (`tools/run_cli.sh` takes `GPU=4` and runs the body under `torchrun`; `accumulation_steps` has
to come down by the same factor to hold `micro_batch_tokens`, and it is a CLI flag). Steady state was
**1.0 s/step against the trunk's 3.29 s/it on one card**, which is the split working. But splitting
the step across ranks changes which examples land in a micro-batch, so the collator emits `(B, L, N)`
triples the single-rank trunk never produced, and each new one pays a full Triton autotune — one was
logged at 330 s for 13 choices. About 45 such stalls per graph cell accounted for **96 % of the wall
clock**, and the flat arm, whose flex kernels are far cheaper to tune, recompiled just as often and
finished its 561 steps in six minutes. The stalls saturate rather than recur, so the cost is bounded
and lands in the shared inductor cache — but four ranks made the graph anneal *slower* than one rank
against a warm cache would have been. Bucketing `B` on the same ladder as `L` and `N` for training
batches, the way `evaluate/scorers.py` already does for evaluation, would make the shape set
independent of the rank count and is the fix worth carrying into the next campaign.

## 8. What arm 2 measured

Six trunks at 5,599 steps, then six `anneal` forks decaying to `lr/10` over 561 steps
(`configs/forks/anneal_molecule_generalist.jsonc`). The annealed checkpoint is the reportable model
and every number below is read off it; the trunk's own scores are milestone measurements, not
results. Run 2026-09-06 at four ranks a cell — the flat anneals took 13 to 18 minutes each, the
graph anneals 2h16 to 2h25.

### Property 1 holds, and holds harder than the trunk showed

Ten relabelings of every test molecule, the margin's spread across them, against the floor the same
run measures (§7, `evaluate/builtin.py`):

| set | graph spread | graph control | flat spread | graph AUROC spread | flat AUROC spread |
|---|---:|---:|---:|---:|---:|
| BACE | 0.2500 | 0.2500 | 6.2083 | 0.0029 | 0.1256 |
| BBBP | 0.2917 | 0.2500 | 11.5833 | 0.0028 | 0.0767 |
| HIV | 0.3333 | 0.2500 | 3.2500 | 0.0109 | 0.1399 |
| SIDER | 0.2917 | 0.2917 | 8.9531 | 0.0030 | 0.0305 |
| Tox21 | 0.2917 | 0.3333 | 5.1667 | 0.0080 | 0.0701 |

**All fifteen graph cells pass**, and the margin's quantum is 0.125 everywhere, so the graph arm sits
at two to three grid steps — the floor, not a signal. The flat arm is 10× to 40× wider on the raw
margin and 10× to 45× wider on the AUROC. The `symmetric` stratum is empty on every task, so nothing
here is diluted by molecules a relabeling cannot move.

The sharper statement is what the anneal did to each arm. The decay grows the margins, and the flat
arm's spread grows with them — BACE 5.17 → 6.21, Tox21 3.46 → 5.17 — exactly as `_control_spread`
describes. The graph arm's did not move off the quantum. The invariance is not merely tight at one
point in training; it survives the margins growing underneath it, on an arm whose comparison does
not.

**The trunk read 11/15 on the same weights, and that was the instrument.** The verdict compared a
spread maximised over ten permutation passes against a control maximised over three, and a maximum
over ten draws of the same noise is larger than a maximum over three — so the control read low, and
read low exactly where the margins were tight enough for one grid step to decide. Every one of the
four trunk failures missed by a single quantum. The control now runs `n_permutations` passes and the
comparison allows one quantum above it, because both sides are maxima of the same grid-valued noise
and a strict `<=` between two of those is a coin flip. What the test has to separate is two orders of
magnitude away.

### Arm 2 − arm 1 on the primary three

Paired within (dataset, seed) against `molecules/PLAN.md` §8.4.8, positive meaning the generalist
beats the specialist:

| set | arm | arm 2 | arm 1 best-val | Δ | arm 1 last-ckpt | Δ (primary) |
|---|---|---:|---:|---:|---:|---:|
| BACE | graph | 0.8185 | 0.8202 | −0.0018 | 0.8133 | **+0.0052** |
| BACE | flat | 0.8667 | 0.8224 | +0.0443 | 0.8338 | **+0.0329** |
| BBBP | graph | 0.7093 | 0.7056 | +0.0037 | 0.6882 | **+0.0211** |
| BBBP | flat | 0.7113 | 0.7157 | −0.0044 | 0.6870 | **+0.0243** |
| HIV | graph | 0.7374 | 0.7691 | −0.0317 | 0.7336 | **+0.0038** |
| HIV | flat | 0.7291 | 0.7617 | −0.0326 | 0.7401 | **−0.0110** |

Pooled over the nine (set, seed) pairs, against the last-checkpoint pairing that §7 makes primary:
graph **+0.0100**, winning 7 of 9; flat **+0.0154**, winning 6 of 9. Against best-val: graph −0.0099
(4/9), flat +0.0024 (5/9).

**The honest reading is that training on every molecule task at once neither helps nor hurts
scaffold-split property prediction at a resolution these seeds can see.** The pooled effect is one to
one-and-a-half ROC-AUC points with a per-pair sd of 0.025 to 0.040, and §8.4.8's own power table puts
three seeds at 0.05 — so this is a null at the resolution available, not a demonstrated gain. It is
also the answer to the question the campaign was built to ask: the free structural labels did not buy
a measurable improvement, and they did not cost one either. The direction is consistent across both
arms and both pairings bar one cell, which is worth more than the magnitude.

### Where the arms actually differ

Within arm 2, averaged over three seeds on the annealed checkpoint:

| category | graph | flat |
|---|---:|---:|
| property classification (5 sets, ROC-AUC) | 0.7795 ± 0.0060 | **0.7969 ± 0.0080** |
| structural probes (9 tasks, EM) | **0.9408 ± 0.0003** | 0.9359 ± 0.0069 |
| `bond_path`, held out | **0.0667 ± 0.0061** | 0.0420 ± 0.0053 |
| `longest_chain`, held out | **0.1013 ± 0.0323** | 0.0460 ± 0.0106 |
| `clintox` FDA_APPROVED, held out | 0.4046 ± 0.0147 | **0.5550 ± 0.0769** |

The split is clean and it runs along one line: **the flat arm wins where pretrained chemistry helps
and the graph arm wins where the answer is a function of the graph.** Property prediction on these
corpora leans on scaffold and functional-group patterns a model that has read a great deal of SMILES
has some purchase on, and the flat arm hands the molecule to those weights in the notation they were
trained in. The graph arm presents a representation the base model has never seen and must learn the
mapping through LoRA and the bias alone. On the structural probes — ring membership, ring size,
functional-group atom membership — and on both held-out structural tasks, where pretraining buys
nothing, the ordering reverses; `longest_chain` by a factor of 2.2.

One number is worth more than its size: the graph arm's probe seed spread is **±0.0003** against the
flat arm's ±0.0069, twenty times tighter. A representation learned from scratch converges to the same
place every time; one that leans on a pretrained prior inherits that prior's seed sensitivity.

This is a mechanism consistent with the data, not one these runs test. Separating it would need an
arm holding both representations, or the same comparison against a base model with no chemistry
pretraining, and neither exists here. Two counter-explanations were tested and discounted: corpus
size does not predict the gap (r = +0.18 between the gap and a task's examples per step, and SIDER is
among the largest and still 0.030 behind), and under-training does not explain it either — the
trunk's worst graph deficit, `ring_count` at 0.698 against 0.859, closed to 0.877 against 0.910
through the anneal, while the property gap survived the decay.

### Generation, with §5's disclosure attached

| | graph | flat |
|---|---:|---:|
| ChEBI-20 METEOR | 0.4573 | 0.4711 |
| ChEBI-20 ROUGE-L | 0.3059 | 0.3146 |
| g2s validity | 0.0560 | 0.1527 |
| g2s `exact_match` | 0.0000 | 0.0193 |
| g2s `roundtrip_match` | 0.0000 | 0.0300 |

On captioning, where both arms face the same task, they perform the same. **The g2s rows are not an
arm comparison and must never be quoted as one** (§5): the flat twin's matched task is
canonicalization, and its input is a randomized SMILES *with stereo* that already spells out every
atom and bond of the answer in the answer's own alphabet. The graph arm generates the string from a
graph. The two numbers measure different tasks.

What the graph column does say on its own terms is that the graph arm emits a valid SMILES about 6 %
of the time and a correct one never, across all three seeds. That is a real capability limit and
belongs in any write-up — as a statement about graph-to-SMILES at this scale and budget, not as a
deficit relative to the flat arm.

**The trunk's generative numbers are not comparable to these.** They were produced by the
right-padded one-row generation path and are understated for the reason §7 gives, so a trunk → anneal
delta on ChEBI-20 or g2s mixes the decay's effect with the padding fix. Only the annealed column
stands alone.

### Property 2, leakage, and the partition

`base_exact` reports `within_tolerance` 1.0 with `max_abs_diff` exactly 0.0 on all six annealed
cells: adapters off reproduces base Llama bit for bit, so the graph machinery is additive and
removable. `leakage` passes on all six — the stereo tag ablation moves `stereo_assigned` by 0.038 to
0.056 against a line of 0.9596, well inside the band. No cell scored a molecule its arm's training
saw in another role.

### What is still owed

A general-text held-out loss, adapter-on against adapter-off, is still not measured, and the
assistant goal makes text ability something this campaign should have reported. The primary claim is
underpowered by design at three seeds; §8.4.8's table says sixteen would be needed for the effect
size actually observed. Both are statements about what the next campaign should carry, not caveats
that change what is written above.

## 9. Optional last step — a larger suite

After the 1B result is in, and only then: the same registry, mixture, partition and harness on
**Llama-3.2-3B** and **Llama-3.1-8B**. Same adapter (`modeling_gtlm_llama.py`), same tokenizer
family, so nothing but the backbone moves. One config each, both arms, one seed first. The
Gemma-3 adapter exists but changes tokenizer and attention layout at the same time, which is a
different experiment.

This is not a result the plan needs. It is worth doing because a small suite of graph-aware
molecule models at three sizes is useful to other people, and because the 1B-vs-larger delta on
the *held-out* tasks is the one place scale could sharpen a transfer claim rather than just a
SMILES-reading one. It does not run until the 1B write-up exists, and it inherits every gate above.
