# The molecule generalist — one model over every molecule task

**Status (2026-09-12):** run and read out. **The reportable campaign is
`configs/probes/008_molecule_generalist_instruct.jsonc`** — six annealed 1B checkpoints, three seeds ×
two arms (graph, and a SMILES flat twin), on `Llama-3.2-1B-Instruct` in chat formatting at the 2×
horizon. That backbone is the one the ladder climbs (`generalist/PLAN.md` D3/D4), so it is the one the
headline numbers come from; §9 Tier 2c is the switch and its read-out.

Everything before it is history and is kept as history: §8 is arm 2 on base weights at 1×, §9 Tier 0.2
the notation ladder, §9 Tier 2 the doubled horizon. Their **property** rows stand. Their **generation**
rows do not — every one of them was measured through the stop-token defect (§8, 2026-09-10) and scores
whether the model stopped rather than whether it was right.

This document is the *what and why*; `DESIGN.md` is the *how*, and its §D9 carries the harness
contracts this campaign settled. **For the notation ladder's results alone**, read
`configs/runs/molecule_generalist.md` — all four notations, one table per family, on base weights.

**Where it sits.** Arm 2 of the molecules plan (§4, "chemistry generalist") and the first consumer of
the generalist harness. Arm 1 is one specialist per task, closed at `molecules/PLAN.md` §8.4.8. Arm 3
— molecules folded into the cross-domain mixture — is the trunk's admission gate and is *not* run
separately; it would duplicate the trunk at the trunk's price.

---

## Summary

One 1B instruction-tuned model, both arms (graph `rich_levi` and the SMILES flat twin), trained on
every molecule task the repo can produce — RDKit structural questions, MoleculeNet property
prediction, ChEBI-20 captioning and graph-to-SMILES generation — with one molecule-level partition
across all sources and three held-out tasks that no training source ever touches.

Four results, in order of the weight they carry.

**The graph arm writes molecules, and writes them better than the flat twin.** Asked for the SMILES of
a molecule it has only ever seen as a graph, it is exactly right **41.9 %** of the time (`validity`
0.756, `roundtrip_match` 0.461) against the flat arm's 21.1 % — and the flat arm's task is
canonicalizing a string that already spells the answer, so this is not an even fight in the graph
arm's favour. By heavy-atom count it is ahead in every bucket and **flat from 10 to 30 heavy atoms**
rather than decaying with size. This is the one place in the campaign where the structural channel has
to survive a *sequence* of decisions instead of reading out at a single position, and it survives.

**The property-prediction claim is a null, and the flat arm's lead is gone.** Five-set mean ROC-AUC:
graph **0.7988**, SMILES **0.7966**. The paired difference is −0.0022 with sd 0.0126 and t(2) = −0.31 —
not significant, and the honest reading is that three seeds do not resolve a gap this size. What
changed against base weights is the sign and the spread: the flat arm's +0.0095 lead became a −0.0022
deficit and the seed spread narrowed rather than growing. The graph arm is ahead on 4 of 5 sets; BACE
is the one where SMILES clearly leads.

**Property 1 is the strong one, and the backbone swap did not touch it.** All fifteen graph cells
(5 sets × 3 seeds) pass on the instruct campaign too, every one of them with its permutation spread at
or below the *control* — two to four grid steps of the margin's quantum, which is the floor the
measurement itself can resolve. §8 is where the comparison lives: on base weights the flat arm's
spread ran 10× to 40× wider and *widened* through the anneal, because its spread grows with its
margins and the graph arm's does not. No amount of pretraining beats a property.

**The structural split is stable across the backbone swap, which is what makes it a finding.** The
graph arm takes `ring_size` by +0.077, `fg_atom_membership` by +0.021, `ring_membership`, and both
*held-out* topology tasks — `longest_chain` by 3.3× and `bond_path` by 1.6× — while the flat arm keeps
`ring_count` and the stereo pair. Every one of those gaps holds its sign and roughly its size from base
weights to instruct. The backbone changed what the model can *say*; it did not change what the
structural channel contributes.

**One earlier result still stands, and it bounds the second.** §9 Tier 0.2 (base weights, 2026-09-07):
writing the molecule as InChI or SELFIES instead of SMILES turns a **+0.0135** flat-minus-graph
property gap into **−0.0138** and **−0.0167**, at the same molecules and the same mixture. So the flat
arm's historical property edge was *SMILES*-specific — an artifact of what the backbone was steeped in
— which is the mechanism the instruct campaign's closed gap is consistent with. Three seeds, and a
truncation artifact pointing the same way as the effect; the caveats are in Tier 0.2 and are not small.

### Checklist

- [x] **Gates.** Arm 1's tuned three-seed closing numbers (`molecules/PLAN.md` §8.4.8, 18/18,
      2026-09-03) — arm 2 is differenced against *these*, never the `lr 3e-5` tables — and the D1–D6
      harness with its plumbing smoke run (`results/BUILD_LOG.md` §T10).
- [x] **Build.** Molecules adapter, the partition, the graph-to-SMILES generator, the ChEBI-20
      loader, the isomeric round-trip test.
- [x] **Cross-check** (2026-09-03, `probes/002` + `003`). BACE seed 0 as a single-task mixture, both
      arms, through this harness: end-of-anneal graph **0.8034**, flat **0.8264**, against §8.4.8's
      three-seed rows (graph 0.8202 sd 0.0120; flat 0.8224 sd 0.0248). Flat lands +0.16 sd from its
      mean; graph lands 1.4 sd low, inside seed noise and the thinner of the two margins, recorded as
      such. The two miss in *opposite* directions, which is worth more than either miss — a trainer
      difference large enough to invalidate arm 2 would push both arms the same way. The arm
      difference, which is what arm 2 reports, agrees with the specialist's seed 0 to 0.002; that is
      one favourable draw rather than a bound, since the specialist's own paired delta ranges −0.025
      to +0.034 across its three seeds. What it establishes is that the harness's delta falls in the
      same range and on the same side as the seed it ran. Five defects and the full write-up in
      `results/BUILD_LOG.md` §T11.
- [x] **Arm 2** (2026-09-06). Three seeds × two arms at 5,599 steps, the §2 mixture, WSD stable
      phase, one anneal fork a cell decaying to `lr/10` over 561 steps. All twelve runs completed;
      the six annealed checkpoints are the reportable models. **82 GPU-h**, measured: a graph cell is
      ~20 (11 h trunk on one card, 9 annealing at four ranks), a flat cell ~7.
- [x] **Read-out and write-up.** §8, with §5's graph-to-SMILES disclosure and §7's generation-path
      caveat attached to the numbers they bear on.
- [x] **The stop-token defect, found and fixed** (2026-09-10, §8). No generative answer in any build
      before this carried a stop token, so `exact_match` scored whether the model stopped. It voided
      the g2s and ChEBI-20 column of every campaign above and it hid the strongest per-task result the
      graph arm has.
- [x] **The instruct backbone** (§9 Tier 2c, run 2026-09-10, read out 2026-09-11). Six cells on
      `Llama-3.2-1B-Instruct` in chat formatting, at `006`'s horizon, mixture and molecules, with the
      stop token on. **These are the reportable checkpoints.** ~80 GPU-h: a graph cell is 8.4–10.2 h
      of trunk on two cards plus its anneal, a flat cell 2.0–2.5 h.
- [ ] **Not measured, and owed** (→ §9): adaptation steps-to-target on the held-out tasks, which
      needs an `adapt` fork per task and none was run; a general-text held-out loss, adapter-on
      against adapter-off.

---

## 1. What goes in

Both arms train on everything below. The graph arm is `rich_levi`, `stereo_tags: on`, no SMILES
anywhere in the prompt — that is what makes graph-to-SMILES a real task rather than a copy. The flat
twin sees SMILES and gets the matched form of each task. Every source is routed by the question text
alone (the question node is `on`, D3), so the model receives the task only through the question.

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
have no anchor ladder (`molecules/PLAN.md` §1), which is a problem for *interpreting* a number against
the field and no problem at all for gradient. Together they are the bulk of the available property
labels, they take arm 2 from three endpoints to about forty, and Tox21 is the only thing in scope that
exercises multi-endpoint routing. Their test AUROC is an internal arm 2 vs arm 1 diagnostic and never
goes in the anchor table. Tox21's ~16k absent labels are skipped at the (molecule, endpoint) level and
the per-endpoint counts go in the run record, because skipping changes each endpoint's effective
weight silently otherwise.

**`stereo_assigned` is the suite's leakage detector** — at chance with the parity channel closed, high
with it open — and it goes in only on a non-degenerate pool. On the `014` pool it is single-answer
(`molecules/PLAN.md` §3.2.10.1) and would be void; the generalist's pool is the train-role union (§3),
so it had to be measured before the mixture was frozen. On build `42f7a14bed21f876` the family is live
here — ten distinct answers on both roles against `014`'s one — but heavily skewed: the drawn test
split answers 0 on 927 of 1,000, so the floor is 0.927, the whole headroom is 7.3 points, the 3σ line
at 500 scored rows lands at 0.962, and only the ~36 rows with a nonzero answer move at all when the
channel closes. `n_stripped` is reported beside the gap for exactly that reason. The family stays in
at its 0.0278 share; what it cannot support is a *fine* reading, so the verdict is a floor test and
nothing more.

It runs as the `leakage` validator (`evaluate/builtin.py`, default set), which closes the channel at
*evaluation* — two scoring passes — instead of training the second model `molecules/configs/016` would
have. The floor is the scored split's own majority-class rate, measured at score time rather than
inherited from `016`'s 0.732, and a single-answer split reports `void` rather than a pass, which is
the one reading that must never look like a clean bill of health. What it does not reproduce is
`016`'s off/on *training* contrast, and that difference is in the safe direction.

**Excluded.** ESOL, FreeSolv and Lipophilicity as *tasks* — the margin readout cannot score a number
and none has an anchor — though their molecules still serve as unlabeled generator pool. QM9, peptides
and text-to-molecule stay out for `molecules/PLAN.md` §1's reasons. A graph arm that *also* sees SMILES
is out of arm 2: it would turn graph-to-SMILES into a copy task and forfeit Property 1. It returns in
§9 as a diagnostic, on those terms and never as the headline model.

## 2. Mixture

Weights are in *examples*, and by D7a each task's gradient share equals its example share (two-level
normalization, per-example within a task).

| Block | share | within the block |
|---|---:|---|
| Tier B (5 sets) | 0.40 | temperature ∝ size^0.5 over the five sets — roughly BACE 5 %, BBBP 6 %, HIV 27 %, Tox21 37 %, SIDER 26 % of the block |
| Tier A (9 families) | 0.25 | uniform over families |
| Tier C (ChEBI-20) | 0.20 | — |
| graph-to-SMILES | 0.15 | — |

**Passes.** Finite sources (Tier B, Tier C) get at most **six**. Generators (Tier A, graph-to-SMILES)
draw fresh examples every pass from the train-role pool, single-pass, so the early-peak overfitting the
specialist runs show cannot come from repetition. The budget is therefore *defined by the finite
sources*, and the registry computes the step count from it rather than taking it as a free knob.

Six and not three, because of how the budget rule interacts with the temperature weighting. The budget
is `min over finite corpora of (passes × train_size) / share`; within Tier B the weight goes as
`size ** 0.5` while the cap goes as `size`, so `available / share` scales as `size ** 0.5` and **the
smallest corpus always sets the horizon**.

| task | share | epochs at 3 | epochs at 6 |
|---|---:|---:|---:|
| `mol/bbbp` | 0.0235 | **3.00** (binds) | **6.00** (binds) |
| `mol/bace` | 0.0203 | 2.67 | 5.35 |
| `mol/chebi20` | 0.2000 | 1.55 | 3.11 |
| `mol/sider` | 0.1036 | 1.07 | 2.14 |
| `mol/hiv` | 0.1062 | 0.52 | 1.04 |
| `mol/tox21` | 0.1465 | 0.43 | 0.86 |

HIV at 0.52 epochs was the sharp end of it: the specialist HIV cell arm 2 is differenced against
trained roughly ten, so a generalist deficit on HIV would have been partly a budget artifact. Six
doubles the horizon to **5,599 steps** and takes HIV just past one epoch. Raising BBBP alone would not
have worked — BACE simply inherits the binding role at 3.00 epochs — so the cap moves for the finite
corpora as a set.

**Six was a ceiling, not the fix**, and the fix now exists: `budget_scale`. A small corpus should be
*drawn less often*, not allowed to end the run when it is exhausted, so the correction belongs in the
sampling weight rather than the cap. `registry.resolve` takes the budget as an input — a multiple of
this rule's own feasible budget — and **down-weights any corpus that cannot sustain its share for that
long**, so BBBP hands over the same 7,464 examples spread across a longer run instead of ending it.

Three properties make it safe to have landed after arm 2 rather than before it:

* **At `budget_scale` 1.0 it is a no-op.** Nothing clamps, the shares are the preset's to the digit,
  and both the config and mixture hashes are unchanged — the twelve cells of §8 and §9 still resolve
  to the digests their run records carry.
* **Redistribution stays inside a block, and where it cannot, the spill is bounded.** The block
  shares — Tier B 40 %, Tier A 25 %, ChEBI 20 %, g2s 15 % — are a statement about what the model is,
  and must not drift far with a compute-budget choice. BACE and BBBP hand their share to
  HIV/Tox21/SIDER; Tier B still holds 40 %. ChEBI-20 is a block of one and has nobody to hand to, so
  it is allowed to shrink and spill the remainder — but no block may fall below **80 %** of its
  design share. Left unbounded the deficit always lands on the generators: at 8× it would take g2s
  from 15 % to 26.5 % and ChEBI from 20 % to 4.8 %, which is a different experiment arrived at by
  choosing a GPU budget.
* **Past the bound it is a refusal naming the block.** ChEBI carries the mixture to 2.41×; BACE's own
  1 % floor binds first at **2.28×**, so ~2× is close to what this mixture genuinely supports. Beyond
  that the budget is bought with an explicit `task_passes` override — in the config and in the hash —
  rather than found.

HIV and Tox21 training under one epoch remains a live alternative explanation for §8's
property-prediction gap, and §9's Tier 2 is what settles it.

**Loss.** Per-example normalization everywhere (D7a default). Captions are 50–100 tokens beside
one-token answers; token-summed loss would make Tier C most of the gradient at a fifth of the
examples.

## 3. The partition — one molecule, one role

The Tier-A generators and graph-to-SMILES draw molecules from the Tier-B corpora. Without a single
rule across sources, a structural question about a BBBP *test* molecule lands in training and the
scaffold split stops meaning "structurally novel". The campaign already had exactly this incident once
(`molecules/PLAN.md` §3.2.10). So:

* **Key:** *stereo-free* canonical SMILES from RDKit, computed once per molecule at adapter build
  time. Stereo-free on purpose: two stereoisomers have identical graphs up to the parity words, so
  keying on the isomeric string would let near-identical graphs straddle the train/test line. Both
  isomers share one role; each keeps its own labels.
* **Roles:** `train`, `val`, `test`, `held_out`. Every molecule in every source gets exactly one.
* **Rule 1.** A molecule in any Tier-B val/test split, any ChEBI-20 val/test split, or anywhere in
  ClinTox is removed from *every* training source. Priority on conflict:
  `held_out > test > val > train`.
* **Rule 2.** Generators draw training molecules only from the `train` role.
* **Rule 3.** Generator *test* sets draw from `test`-role molecules, scaffold-novel by construction.
* **Rule 4.** The registry refuses to build a mixture violating rules 1–3, and the run record carries
  the per-role molecule counts and the number of cross-source overlaps removed.

Enforced in `adapters/molecules.py`, pinned by a test that rebuilds the partition from the raw CSVs
and asserts pairwise disjointness (`DESIGN.md` §T2).

**The `val` role is larger than it needs to be.** Rule 1 removes every val-role molecule from *every*
training source, so the 7,690 molecules in that role cost training data across the whole mixture, and
they buy a diagnostic that never selects anything — WSD has no dev-score selection, the reportable
model is the end of the anneal, and §8.4 measured Tier-B validation *anti-ranking* the arms on BBBP. A
few hundred rows per source would read the same curves. It was left alone here because the partition
feeds `build_version` and shrinking it means a full rebuild; it folds into §9's Tier 2 rebuild
alongside the sampling-weight fix.

## 4. Held out

| Held out | Why this one | Scored as |
|---|---|---|
| `bond_path` | Declared 2026-08-28 (`molecules/PLAN.md` §4.1). SPD *is* the answer by construction, and SPD is the graph arm's bias, so it is the cleanest test of whether the structural channel crosses question templates. | zero-shot exact match, then steps-to-target from the generalist vs from base Llama |
| `longest_chain` | Added 2026-09-02. With `bond_path` it makes the held-out set *the traversal family* while training covers rings, functional groups and stereo. `ring_count` was the alternative and is weaker, because `ring_size` and `ring_membership` are in training and transfer there is near-duplicate. It stays measured as a specialist (`014`: graph 0.988 vs flat 0.828), so nothing is lost by holding it out. | same two ways |
| ClinTox | Declared 2026-08-28. A toxicity / trial-failure endpoint, unlike binding or permeability. | zero-shot AUROC, then steps-to-target |

Two Tier-A holdouts is the number. A third starts costing training coverage for a declaration made
after seeing results, which is worth less than the two made before.

Zero-shot is measured (§8). **Steps-to-target is not** — it needs three held-out tasks × two starting
points (the generalist and base Llama) × three seeds, eighteen short `adapt` forks, and none was run.
Three seeds is not padding: a first-crossing statistic is noisier than an end-of-run score, and one
seed cannot separate a real gap from where the curve happened to cross. It is Tier 1 work in §9.

Few-shot here means *few-example fine-tuning*, not in-context examples. Several molecule graphs in one
prompt is not what the prefix-node layout is built for, and the in-context anchor in the Tier-B table
(Vicuna-13B, 4-shot) sits at chance anyway.

The molecules package refuses to build `bond_path` and ClinTox without `held_out_eval`
(`HELD_OUT_TIER_A_TASKS`, `HELD_OUT_DATASETS` in `data.py`); `longest_chain` joined those tuples, and
the generalist registry mirrors all three, so a mixture naming any of them fails in both places.

## 5. Graph-to-SMILES

The one task not in the molecules plan. The graph is on the *input* side, so it is the inverse of the
text-to-molecule generation `molecules/PLAN.md` §1 excludes, and it is the bridge to captioning: a
model that can write a molecule's structure is better placed to describe it.

* **The target is RDKit canonical SMILES**, since the task is not one-to-one and canonicalization
  makes it a function of the molecule. Three metrics in order of what matters: validity (RDKit
  parses), round-trip match (parse, canonicalize, compare), canonical exact match (strict, and it
  additionally asks the model to reproduce an RDKit atom ranking).
* **Stereo is out of the target, for a structural reason rather than as a shortcut.** The node text
  carries the tetrahedral parity word (`cw`/`ccw`) and the bond stereo word (`E`/`Z`), but a parity
  word is only meaningful relative to a neighbour *ordering* and the graph has none — that is what
  permutation invariance means. A graph arm asked for `@`/`@@` would be asked for information it does
  not have. So the target for **both arms** is `Chem.MolToSmiles(mol, isomericSmiles=False)`, which
  keeps the comparison matched: the flat twin's input carries stereo it must learn to *drop*, the
  graph arm's input carries parity words it must learn to *ignore*. Emitting a stereo mark is an error
  under this target and is recorded as a diagnostic. E/Z is in principle recoverable from connectivity
  through CIP priorities and could be added back once the round-trip test reconstructs it; tetrahedral
  chirality would need an order-independent parity encoding, which is an encoding decision for the
  molecules plan rather than this document.
* **The flat twin's matched task is canonicalization** — randomized SMILES in, canonical out. The
  graph arm has no input order to randomize, so it faces the hard version by construction. **The two
  columns therefore measure different tasks and must never be quoted as an arm comparison** (§8).
* It is a generator with free labels, capped at one example per train-role molecule per pass.
* Question text: `Question: write the canonical SMILES for this molecule.` No atom labels.

## 6. ChEBI-20

* Keeps its own split (26,407 / 3,301 / 3,300) and folds into the §3 partition. Overlap with
  MoleculeNet is expected to be small and is *measured*, not assumed.
* Per-example loss, as §2.
* ChEBI-20 includes salts and multi-fragment molecules. Disconnected graphs put SPD at the `max_spd`
  clamp between components, and larger molecules hit the node budget. Both are checked at build time;
  the heavy-atom cap is chosen against the ChEBI size distribution and recorded.
* The templated-caption caveat from `molecules/PLAN.md` §1 travels with every Tier-C number: BLEU and
  ROUGE reward template matching, so a strong number is weak evidence.

## 7. Recipe, measurement, and what gets reported

**Recipe.** Both arms at `lora_r 16` (the r32 axis is closed, §8.4.5), `lora_dropout 0.05` (the
molecules value, so arm 1 and arm 2 match; the trunk's D3 value of 0.15 is not used here),
`bias_lr 1e-2` on the graph arm, `weight_decay 0.1`, `max_spd 32` (`molecules/PLAN.md` §8.4.6,
ablation §8.4.9).

**The learning rate is `1e-4`, matched across arms — settled 2026-09-04.** The specialist settled it
per (task, arm), 3e-4 on BACE and BBBP against 1e-4 on HIV, and one mixture cannot hold both. It
inherits the lower one for two reasons. **The schedule is not the one those rates were tuned on:**
the specialists ran warmup + cosine, which touches its peak for a moment, while WSD holds the stable
phase at `lr` for essentially the whole run — the same number is a materially larger dose here, and
carrying a cosine peak across as a constant rate would be §3.2.7 and §8.4.4's mistake in a third
place. **And the risk is asymmetric:** 3e-4 was measured to cost the graph arm 0.109 ROC-AUC on HIV
(§8.4.7), where it also produced the screen's highest validation score and its lowest test score,
while 1e-4 on BACE and BBBP came out worse rather than broken. HIV is 10.6 % of this mixture against
BACE's 2.0 % and BBBP's 2.4 %. `lr_min` follows to `1e-5`, keeping the anneal at `lr/10`.

**Schedule:** WSD — short warmup, constant stable phase for the §2 budget, one anneal fork decaying to
`lr/10` over ~10 % of the stable steps. The annealed checkpoint is the reportable model. There is **no
test-set selection and no best-val selection**: Tier-B validation anti-ranks arms on BBBP (§8.4) and
the anneal fork makes selection unnecessary.

**The comparison.** Arm 1 records both its best-val and its last-checkpoint test score. Arm 2 minus
arm 1 is reported against *both*, paired within (dataset, seed), with the last-checkpoint pairing
primary because it is the one free of a selection instrument shown to be near-blind.

**Two claims, never conflated** (`molecules/PLAN.md` §8.4.3.1): mean ± s.e. over seeds on the fixed
test set, which is what the anchors publish; and the paired per-molecule bootstrap, which is the
generalisation claim. HIV's effective n is ~132 actives, not 4,112 molecules.

**Disclosures that travel with every number:** Tox21 / SIDER are not anchor-comparable; the Tier-C
caveat; the flat twin's graph-to-SMILES is canonicalization; the partition counts.

**The six cells.** The `molecule_generalist_{graph,flat}_s{0,1,2}` cells of
`configs/runs/molecule_generalist.jsonc`, launched one at a time through `tools/chain.sh <file>
<cell>`. The recipe is written once at the top of that file and the cells differ only in what its
bundle and seed axis vary, so a recipe decision has one place to be corrected;
`test_the_campaign_cells_differ_only_where_they_are_meant_to` asserts that every field outside that
list agrees across all of them. All six read one build, `42f7a14bed21f876`, so a difference between
two cells is the run's own spread and nothing else.

**Resolved budget:** **5,599 steps** on both arms — 318,217 examples on the graph arm, 318,179 on the
flat one. Bound by `mol/bbbp` at its six-pass cap (§2). `max_steps` pins the horizon on all six cells
rather than leaving it to the budget rule, for the reason below.

**The arms are matched in examples, not tokens, and that costs a second number.** D4.4 sets the batch
in tokens so a mixture of very differently-sized tasks costs a roughly constant amount per step. That
is the right default and exactly wrong for an arm comparison: the built graph mixture measures 288.28
tokens an example against the flat arm's 82.51, so one shared token budget would hand the flat arm
3.5× the batch. The graph arm runs at `tokens_per_step 16384` and the flat arm at **4689**, the value
landing on the same ~56.83 examples/step (`tools/tokens_per_step.py`). No integer token count lands
the flat arm on 5,599 steps as well — 4689 resolves to 5,600 and 4690 to 5,598 — so `max_steps` pins
5,599 for both and the flat arm takes the value on the *short* side. The pair matches exactly on
schedule length and differs by 0.012 % on examples per step, which is the right way round: a step
count is what the WSD phases are measured in. **Any new arm re-derives its own `tokens_per_step`**
rather than inheriting one, since a change to the input changes the tokens-per-example.

**Three harness facts bear on how these numbers read.** The rest is in `DESIGN.md` §D9, with the
narrative in `results/BUILD_LOG.md` §T12.

1. **Generative metrics measured before 2026-09-06 are understated and not comparable with these.**
   Generation ran right-padded, and `prepare_inputs_for_generation` then numbered every continuation
   from 1, placing it under RoPE *before* the prompt it answered. Teacher-forced numbers were
   unaffected. So a trunk → anneal delta on ChEBI-20 or g2s mixes the decay's effect with the padding
   fix, and only the annealed column stands alone.
2. **Evaluation is batched and sharded**, 8–11× and a further 3.4× at four ranks, metric-identical
   (889/889 keys). A milestone that cost 133 minutes costs about 12 on one card. Evaluation cost is no
   longer a reason to thin what gets scored.
3. **Four ranks made the graph anneal slower than one would have**, because splitting a step changes
   which examples share a micro-batch and every new `(B, L, N)` triple pays a Triton autotune — 96 %
   of that leg's wall clock. Bounded, cached, and the reason §9 prices graph cells at one card.

## 8. What arm 2 measured

**This section is history.** It is arm 2 on base `Llama-3.2-1B` in `Q:/A:` formatting at the 1×
horizon, kept because it is where Property 1, the arm split and both defects were established. The
reportable campaign is §9 Tier 2c. Everything generative below was measured through the stop-token
defect this section goes on to document, and is a floor rather than a score.

Six trunks at 5,599 steps, then six `anneal` forks decaying to `lr/10` over 561 steps
(`configs/forks/anneal_molecule_generalist.jsonc`). The annealed checkpoint is the reportable model
and every number below is read off it; the trunk's own scores are milestone measurements, not results.
Run 2026-09-06 at four ranks a cell — the flat anneals took 13 to 18 minutes each, the graph anneals
2h16 to 2h25.

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
point in training; it survives the margins growing underneath it, on an arm whose comparison does not.

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

The split is clean and it runs along one line: **the flat arm wins where pretrained SMILES helps and
the graph arm wins where the answer is a function of the graph.** Property prediction on these corpora
leans on scaffold and functional-group patterns a model that has read a great deal of SMILES has some
purchase on, and the flat arm hands the molecule to those weights in the notation they were trained
in. On the structural probes — ring membership, ring size, functional-group atom membership — and on
both held-out structural tasks, where that notation buys nothing, the ordering reverses;
`longest_chain` by a factor of 2.2.

**The asymmetry is narrower than "one arm has a prior and the other does not", and stating it loosely
would not survive a check.** The graph arm's node text is English — `carbon aromatic ring deg2 H1`,
`single`, `chiral cw` (`data.py:72`) — ordinary high-frequency tokens used in their ordinary sense, so
the backbone reads it perfectly well and brings a general chemistry vocabulary to it. What the graph
arm lacks is not chemistry knowledge; it is exposure to *its own notation as a molecular
serialization*, and a structural channel the base model has never had in any form. The flat arm's
advantage, if it is one, is specifically SMILES-string statistics: scaffold-level regularities over a
compact notation that appears in text beside statements about what molecules do.

One number is worth more than its size: the graph arm's probe seed spread is **±0.0003** against the
flat arm's ±0.0069, twenty times tighter. A representation learned from scratch converges to the same
place every time; one that leans on a pretrained prior inherits that prior's seed sensitivity.

**This is a mechanism consistent with the data, not one these runs test.** Separating it needs the
same comparison run at more than one level of notation exposure, or an arm holding both
representations — neither exists here, and both are §9. Two counter-explanations were tested and
discounted: corpus size
does not predict the gap (r = +0.18 between the gap and a task's examples per step, and SIDER is among
the largest and still 0.030 behind), and under-training does not explain it either — the trunk's worst
graph deficit, `ring_count` at 0.698 against 0.859, closed to 0.877 against 0.910 through the anneal,
while the property gap survived the decay. That second discount is bounded: it rules out the graph arm
being *globally* behind on training, not the property sets specifically being short (§2 — HIV and
Tox21 train under one epoch), which is why §9 keeps a horizon experiment.

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
graph.

**Every g2s number in that table is void.** It was read for two months as a capability limit — the
graph arm emits a valid SMILES about 6 % of the time and a correct one never — and it is not one. The
built data carried no stop token, so no model trained here was ever shown where a string ends, and on
the two kinds that are scored by *generating* that is fatal. See "A stop-token defect found on
2026-09-10" below for what the graph arm was actually doing. The ChEBI-20 rows are measured through
the same defect and are understated by an unknown amount on both arms.

### A truncation defect found on 2026-09-06, and what it touches here

**SIDER's flat number is measured over ~4 % corrupted rows, and the graph arm's is not.** A flat arm
is a single node, so its whole prompt is one text and `max_length` (512) truncates it *from the
right* — taking the trailing `A: Yes` with it. `render` then supervises "the prompt node's last
token", which on a truncated row is a mid-molecule token: `sider/flat` row 324 is supervised on
`@@` against a stored answer of ` Yes`. The margin readout duly reports `y_true` = No and reads the
margin at a position that means nothing. Nothing raises.

Truncated rows in each test split, out of the split's total:

| | SMILES | SELFIES | InChI | graph |
|---|---:|---:|---:|---:|
| SIDER (3,861) | **162** | 297 | 270 | **0** |
| Tox21 (7,148) | 0 | 49 | 0 | 0 |
| HIV (4,112) | 0 | 4 | 1 | 0 |
| BBBP (204) | 0 | 1 | 0 | 0 |
| BACE (152) | 0 | 0 | 0 | 0 |

**The graph arm is immune**, and that is the uncomfortable part rather than a reassurance:
`max_length` is *per node*, and an atom text is a handful of tokens, so no graph row is ever
truncated. The defect therefore lands on one arm of a two-arm comparison. Of the five property sets
only SIDER is affected on the SMILES arm — BACE, BBBP and HIV peak at 151, 204 and 185 tokens — so
what it touches is SIDER's contribution to §8's five-set average (flat 0.7969 against graph 0.7795).
Label noise pushes an AUROC toward 0.5, so the direction is that the flat arm's SIDER is *understated*
and the flat lead is if anything larger; the magnitude is not quantified here, because §8 records the
five-set average rather than per-set trained numbers, and re-scoring those cells is the only way to
put a number on it.

**What it does to *training* is the half that cannot be repaired by re-scoring, and it is small.**
Evaluation can exclude a truncated row from every arm and does; training swallowed it. Measured over
the whole training mixture by `tools/truncation_census.py`, weighted by each task's share:

| | graph | SMILES | SELFIES | InChI |
|---|---:|---:|---:|---:|
| share of the training draw truncated | **0.000 %** | 0.165 % | 0.521 % | 0.332 % |

SIDER is essentially all of it (189 / 15,390 rows on SMILES, 572 on SELFIES, 405 on InChI). ChEBI-20
and g2s were the two sources worth suspecting, because their answers are a full caption and a full
SMILES string competing with the molecule for the same 512 tokens — measured, they are clean (26 /
20,477 and 10 / 4,000, both SELFIES only). The four Tier-A families held at SMILES on every arm
truncate identically across the flat arms and so cannot differentiate them.

A ~0.5 % label-noise differential, concentrated in one of five sets, is far too small to account for
§9's 0.0167 reversal. **The defect is a disclosure, not a reason to rebuild at this size** — which is
worth stating explicitly, because the opposite conclusion was reached once before the census existed.

How it was found is the reusable part. `pos_rate` is `y_true.mean()` over the scored rows, and the
arms score identical molecules — verified key for key and answer for answer across all four builds —
so it is a property of the task and cannot differ by arm. It differed: SIDER read 0.5460 on the graph
arm against 0.5180 and 0.5080 on two flat ones. That is the whole defect in one number, and it was
visible in every run that ever printed it. It is now asserted rather than printed
(`tools/notation_probe.py::check_arms_agree_on_labels`), with a test that reproduces those three
values — `molecules/PLAN.md` §9's rule, which this campaign has now paid for a fifth time: a quantity
that is only ever *read* has no error-detecting surface.

### A stop-token defect found on 2026-09-10, and the g2s column it voids

**No generative answer in this campaign ever carried an end-of-text token, so the graph arm's
graph-to-SMILES score measures whether it stopped, not whether it wrote the molecule.**

`schema.render` supervises the answer's tokens and nothing after them —
`TextGraphDataset.tokenize` has always taken an `add_eos` flag and the adapter never passed it. On
`token` and `yesno` that is invisible: the readout is a logit margin at a known position and
generation never runs. On `text` and `smiles` the model has to decide where to stop and was never
shown an example of stopping.

What the graph arm is actually doing, over the whole 1,000-molecule g2s test split at the 2× anneal
(`tools/g2s_report.py`, seed 0):

| | graph | flat |
|---|---:|---:|
| `exact_match` | 0.0000 | 0.0740 |
| **target is a prefix of the prediction** | **0.4650** | 0.2100 |
| prediction over 2× the target's length | 0.9870 | 0.3350 |
| mean prediction characters (target 44.8) | 365.2 | 142.3 |

The graph arm writes the exactly-correct canonical SMILES and then keeps writing, usually a repeating
fragment until the 256-token generation cap:

```
target      CC(=O)C(O)N1C(=O)C=CC1=O
prediction  CC(=O)C(O)N1C(=O)C=CC1=OBr. CC(=O)O1C(=O)C=CC1=OBr.[CH3][NaH]1[CH3]C1=O.[CH3]C1=O. …
```

Three things follow, and the first is the one that matters.

* **The graph arm can serialize a graph.** A 46.5 % prefix rate is a *floor* on its real
  `exact_match`, earned with 15 % of a sixteen-task mixture and 2.2 epochs. Not chance: on targets of
  ten characters or more it is 0.4632, and a canonical 45-character string does not become a prefix
  by accident. §8's "a correct one never" was an artifact.
* **It also reverses the arm reading.** The graph arm's prefix rate is more than twice the flat
  arm's (0.4650 against 0.2100). §5's rule still stands — the flat twin is doing canonicalization and
  the two columns are not an arm comparison — but the direction of the number that *was* being quoted
  is not the direction of the underlying capability.
* **The defect is not arm-neutral, so it flattered the flat arm.** A randomized SMILES input is about
  as long as the answer, which is a length cue the graph arm has nothing to match: 33.5 % runaway
  against 98.7 %. Any generative comparison measured under it is confounded in the flat arm's favour.

**Which stop token, checked rather than assumed.** Every config here names `meta-llama/Llama-3.2-1B`
— the *base* weights, not the instruct variant (`molecules/PLAN.md` §8.4.4: an instruct base would
confound what the prefix nodes contribute). A Llama-3 vocabulary carries both `<|end_of_text|>`
(128001) and `<|eot_id|>` (128009) whichever variant it came from, so the vocabulary does not say
which to use; the checkpoint's `eos_token_id` does, and on base weights it is 128001. Two more things
had to agree, and `tools/stop_token_check.py` now asserts all three on any backbone:

| id | token | embedding row norm | ×ordinary token |
|---|---|---:|---:|
| 128001 | `<\|end_of_text\|>` | 0.9683 | **0.980** |
| 128009 | `<\|eot_id\|>` | 0.5449 | 0.552 |
| 128002 | `<\|reserved_special_token_0\|>` | 0.5450 | 0.552 |
| 128008 | `<\|eom_id\|>` | 0.5450 | 0.552 |
| 128010 | `<\|python_tag\|>` | 0.5450 | 0.552 |

`<|eot_id|>` sits at the reserved block's value to four decimals — that is initialisation, not
training. `tie_word_embeddings` is true and the LoRA recipe targets the attention and MLP projections
only, so **the output head is frozen**: a token whose row was never trained cannot be emitted however
much the adapters want it. Supervising `<|eot_id|>` on base weights would have reproduced this defect
instead of fixing it. `generation_config.eos_token_id` is `[128001]`, so it is also the token
`generate` stops on. On an Instruct backbone all three answers move together to 128009 — including
the trained-row check, since ending a turn is what an instruct checkpoint does — which is why the
build reads `tokenizer.eos_token_id` rather than pinning a number.

**This is not the trunk's answer, only this campaign's.** `generalist/PLAN.md` D4 leans
Llama-3.1-8B-Instruct for the 7–12B trunk, and D3 — decided — pairs instruct weights with chat
formatting, "both together, neither alone". Retargeting the stop token is automatic; reconciling it
with a chat template is not, because the template closes the assistant turn with that same token and
appending on top of it writes it twice. `schema.render` carries the note, since that is where D3
lands and where the answer boundary stops being `\nA:`.

Fixed the same day: `answer_eos` (default on) appends and supervises that token on
`GENERATIVE_ANSWER_KINDS`, within `max_length`. It moves `build_version`, so the configs that
reproduce a published number pin it to `false` and the campaign's artifacts stay reachable —
`test_the_arm2_build_version_does_not_move` asserts exactly that, and
`test_the_stop_token_is_a_different_build` asserts the fix cannot land silently inside the build §8
reports.

**What is not yet known** is where the fixed number lands. The prefix rate is a floor read off a
model trained without the token; it is not the same thing as a model trained with it. §9 Tier 2b is
the run that answers it.

How it was found is the reusable part, and it is `molecules/PLAN.md` §9 again in a new costume: a
rate has no error-detecting surface either. `exact_match 0.0000` is the same number whether the model
emits nothing, emits a plausible wrong molecule, or emits the right one and runs on — and it was
reported, re-reported at 2× and quoted in three tables before anyone looked at a prediction.
`tools/g2s_report.py` exists so that the next zero comes with its predictions attached and a
heavy-atom ladder under it.

### Property 2, leakage, and the partition

`base_exact` reports `within_tolerance` 1.0 with `max_abs_diff` exactly 0.0 on all six annealed cells:
adapters off reproduces base Llama bit for bit, so the graph machinery is additive and removable.
`leakage` passes on all six — the stereo tag ablation moves `stereo_assigned` by 0.038 to 0.056
against a line of 0.9596, well inside the band. No cell scored a molecule its arm's training saw in
another role.

## 9. What runs next

**The decision, 2026-09-06: mechanism before scale.** Arm 2 produced two nulls and one differentiated
result, and the differentiated one is the only thing in the campaign with a mechanism attached that
has never been tested. Nothing below adds seeds to a null; everything below either tests §8's
mechanism or removes a known artifact from under it.

**Why the backbone ladder is not first**, against `molecules/PLAN.md` §10.1, which is written and
config-complete:

1. **Its motivation is unsourced.** §10.1 carries its own warning — the sign-flip-with-backbone-scale
   observation "exists as a recollection of a separate experiment" and is recorded nowhere in this
   repository. Committing ~170 GPU-h at 3B to chase a number nobody can point at is the shape
   `molecules/PLAN.md` §0 exists to avoid.
2. **§0 predicts it should not flip here.** The headline mechanism is preserving node text, and an
   atom is `C`. Scaling a backbone does not create node text, so the ladder's most likely outcome on
   molecules is a third null.
3. **The budget rule under it is known-broken.** §2 calls the six-pass cap a ceiling rather than a
   fix, and HIV and Tox21 still train under one epoch. Scaling the backbone first bakes that artifact
   into runs costing two to four times as much, and it is a live alternative explanation for the very
   gap the ladder would be asked to interpret.

**Costs, and arm 2's are not the ones to plan with.** Arm 2 measured ~20 GPU-h for a graph cell (11 h
trunk on one card, 9 annealing at four ranks) and ~7 for a flat one, ~82 for the six-cell campaign —
and **those numbers are obsolete in the direction that matters.** They were paid before the
evaluation fixes of `DESIGN.md` §D9, when a single milestone cost 133 minutes on the flat arm against
34 minutes of training; most of arm 2's wall clock was evaluation, not gradient. Measured after the
fix, on the Tier 0.2 campaign of 2026-09-07: a notation cell is **~2.8 GPU-h** — 5,599 steps in about
62 minutes at two ranks *including* both milestones, plus ~20 minutes annealing — and all six cells
ran concurrently on twelve B200s in **1h27 wall, ~18 GPU-h total**. Planning a flat-class cell at 7
GPU-h now overstates it by about 2.5×.

Two things carried that: batched, sharded evaluation (§D9), and a warm inductor cache shared with
arm 2's runs. Scoring a trained checkpoint through `tools/notation_probe.py --checkpoint` is another
~0.2 GPU-h a cell.

Dataset builds are CPU-only and are now the long pole rather than training: both notation arms took
**73 minutes** for the full mixture, and it over-built — `data_prep` run under a config whose
`tokens_per_step` is not the one the arm will train at computes a different step count and therefore
more generator passes than the run needs. Harmless (a superset) but worth an hour; point `data_prep`
at the arm's own config. A new `arm` value is part of the source cache key, so any arm that changes
the input rebuilds.

### Tier 0 — is the flat arm's edge a pretraining prior? (~50 GPU-h)

§8 attributes the flat arm's 0.0174 property-prediction lead to a SMILES prior the graph arm cannot
reach. Either that is true, in which case the campaign's story stops being "the arms tie" and becomes
*the arms tie because the flat arm borrows a prior the graph arm has no access to, and the graph arm
wins everywhere that prior is worthless* — or it is false and the lead is architectural. Both are
worth more than another seed.

**The instrument is the notation, not a corrupted input.** SMILES, SELFIES and InChI all determine the
same molecule, so a difference between them cannot be an information difference; what separates them
is how much of each the backbone read during pretraining. That makes the flat arm's notation a real,
meaningful axis along which the pretraining prior varies, with every input a string a chemist could
read. An earlier draft of this tier proposed scrambling the SMILES alphabet instead; it is dropped,
because a corrupted notation measures behaviour off-distribution and a drop there is as easily
"confused by nonsense" as "lost a prior".

**0.1 — Zero-shot AUROC across the three notations, and the graph arm's floor.** ~3 GPU-h, no
training. Base Llama scored through the same yes/no margin readout on the five property sets, once per
notation, plus the graph arm with **adapters off** — which `base_exact` proves is bit-exact base
Llama, so it reads node text with no structural bias and nothing trained. Report tokens-per-example
beside every row.

What this establishes is a *floor ordering*, and the honest version of the claim it supports is:
the graph arm starts from a general-language prior over chemistry vocabulary and no structural prior
at all, while all three flat notations start from a notation-specific prior on top of that — and the
weakest of the three still starts ahead. Note that "the graph arm has no prior" is **not** available
and must not be written: its node text is English (`carbon aromatic ring deg2 H1`), which the backbone
reads perfectly well, and a zero-shot graph score above chance is the expected outcome rather than a
problem.

**And the graph row is the floor for the graph *arm*, not a reading of what the backbone knows.** It
conflates two things that are both genuinely part of what the arm starts from — no structural bias,
and an attention layout (per-node RoPE reset, bidirectional prefix) the backbone was never pretrained
under. That makes it the right quantity for "where does this arm begin", and the wrong quantity for
"how much chemistry does Llama have"; the latter would need the node text as ordinary concatenated
prose, which is a different measurement and is not this one.

**Pre-registered, because it changes what 0.2 can claim.** All three notations may well read at chance
on the property sets — §4's in-context anchor (Vicuna-13B, 4-shot) already does, and §8 shows these
tasks are learned in training rather than elicited. If that happens, 0.1 yields **no measured
ranking**, and the exposure ordering used in 0.2 rests on the corpus-frequency argument — SMILES is
vastly more common in text than SELFIES or InChI — rather than on anything this probe established.
That is defensible and it is not a measurement; the write-up says which one it is leaning on.

#### 0.1 RESULT — 2026-09-06: no arm has a useful prior, and SMILES is the odd one out

`tools/notation_probe.py`, `results/notation_probe/probe.json`. Base Llama-3.2-1B, no bias, no
adapter, nothing trained; the §8 margin readout and the same seeded 500-row subsample, minus the rows
excluded below.

| set | n | SMILES | InChI | SELFIES | graph |
|---|---:|---:|---:|---:|---:|
| BACE | 152 | 0.2649 | 0.4420 | 0.5410 | 0.5587 |
| BBBP | 203 | 0.5456 | 0.4271 | 0.4769 | 0.4577 |
| HIV | 500 | 0.3294 | 0.5323 | 0.6624 | 0.6343 |
| Tox21 | 496 | 0.4389 | 0.4818 | 0.3869 | 0.4352 |
| SIDER | 459 | 0.4752 | 0.4732 | 0.5219 | 0.5325 |
| **mean** | | **0.4108** | 0.4713 | 0.5178 | 0.5237 |
| **mean \|AUROC − 0.5\|** | | **0.1074** | 0.0416 | 0.0723 | 0.0665 |

**The pre-registered outcome came true: there is no ranking here to select on.** Nothing is
consistently above chance. Every arm's mean sits within 0.09 of 0.5, no arm wins more than three of
the five sets, and the per-set spread inside one arm (SMILES runs 0.26 to 0.55) dwarfs the spread
between arms. Whatever the flat arm's trained advantage is made of, it is **not** knowledge that can
be read out of the backbone at zero shot — which sharpens §8's claim from "pretrained chemistry" to
something about fine-tuning efficiency, and is the more precise statement.

**SMILES is the outlier, and in the direction nobody proposed.** It has the *lowest* mean AUROC
(0.4108) and simultaneously the *most* signal of the four (mean deviation 0.1074, against 0.0416 for
InChI), because its errors are systematic rather than random: 0.2649 on BACE and 0.3294 on HIV are
strongly **anti**-predictive. The backbone has a real, consistent disposition about SMILES strings
and it points the wrong way. This is why the ladder is reported and not collapsed to a pick: "lowest
zero-shot performance" and "least pretraining prior" turn out to be *opposite* orderings on this
data — SMILES scores lowest and carries the most prior, InChI sits nearest chance and carries the
least — so a selection rule phrased on raw score would have chosen SMILES precisely because the
backbone knows it best.

**The graph arm's floor is not zero, as expected.** 0.5237 mean, 0.6343 on HIV — the backbone reads
`carbon aromatic ring deg2 H1` perfectly well. What it does not have is a structural channel, and the
row confirms the framing this section already required: the honest sentence is that the graph arm
starts with a general-language prior and no structural one, not that it starts with nothing.

One instrument note that bears on reading the table: the flat arms tie 13–35 % of pairs (SELFIES on
BACE ties 34.8 %, so that cell is the weakest in the table) while the graph arm ties 3–4 %. An
untrained model emits a much coarser margin on a single-node prompt than on a multi-node one, so the
flat cells are noisier than the graph cell at equal n.

**Excluded rows, dropped from every arm and reported rather than silently absorbed:** SIDER 297,
Tox21 49, HIV 4, BBBP 1, BACE 0 — all truncation, zero un-encodable. The first run of this probe was
made *without* those exclusions and is superseded; it read SIDER as high as 0.6066 where the corrected
number is 0.5219, which is the size of the defect and the reason §8's SIDER caveat exists.

**0.2 — Two more trained flat arms, and the deliverable is the gradient.** SELFIES and InChI, the same
recipe and mixture as arm 2, three seeds each: ~42 GPU-h, or ~14 for one seed a notation first. The
SMILES leg already exists, which is why this is affordable.

**All three are reported, and none is selected.** Choosing the notation with the lowest base score and
comparing only against that is baseline-shopping — the same error as §8.4.6's tie-break rule exists to
prevent, and it would make any outcome unquotable. SMILES stays the headline baseline for any
competitive claim; the *slope* across the three carries the mechanism claim. This is §8.4.3.1's two
claims, never conflated, applied to this question.

| | zero-shot AUROC | tokens/example | trained flat | flat − graph |
|---|---|---|---|---|
| SMILES | | 9 | 0.7969 | +0.0174 |
| SELFIES | | ~37 | ? | ? |
| InChI | | ~49 | ? | ? |
| graph (adapters off, then trained) | | 288 | 0.7795 | — |

Monotone shrinkage down the last column as exposure falls → **the mechanism holds**. Last column flat
while the notations differ → **it does not**, and the flat arm's edge is architectural. Either reading
is a result, and neither depends on having picked a favourable baseline.

**Equally expressive is not equally easy, and it is a competing cause.** Measured on
`CNC(=O)c1ccccc1` through the Llama tokenizer: SMILES 9 tokens, SELFIES ~37, InChI 49 — four to five
times the sequence for the same information. A notation can therefore score lower because the backbone
read less of it (what this tier isolates) or because it is longer and more fragmented to attend over
(what it does not). Tokens-per-example is a disclosure on every row, and §7's rule binds: each
notation re-derives its own `tokens_per_step`, or the short-sequence arm is silently handed a bigger
batch. Two further disclosures: InChI merges tautomers, so it is not strictly bijective with the
molecule the way SELFIES is, and its formula layer (`C8H9NO`) hands over the composition that SMILES
makes the model derive.

**Built 2026-09-06, and two things had to be decided to build it.** The notation rides on the *arm*
(`adapters/molecules.py` `FLAT_NOTATIONS`: `flat`, `flat_selfies`, `flat_inchi`) rather than on
`MoleculeAdapterConfig`, because `build_version` hashes every field of that config and a `notation`
field there would have invalidated `42f7a14bed21f876` — the build all six arm-2 cells read. `arm` is
already part of `source_path`, so each notation gets its own artifact inside the same build, and §8's
data stays valid. A test asserts the hash has not moved.

**SELFIES needed a constraint decision, and it is a chemistry decision.** SELFIES' semantic
constraints are what make its guarantee true — every string decodes to a valid molecule — so they are
a valence table, and the stock presets refuse part of this pool. Measured over all 53,921 Tier-B
molecules: `octet_rule` fails to encode 6,863 (12.7 %), `default` 40, `hypervalent` 22. Every one of
the 22 is an organometallic — ferrocenes, molybdenum and tungsten carbonyls — whose metal centre
carries nine or ten bonds against the preset's catch-all cap of eight, and one of them is test-role.
The settled set is `hypervalent` with the catch-all raised to 12, which encodes **all 53,921 and
round-trips every one of them**. That is recorded in `SELFIES_CONSTRAINTS` and in the probe's run
record, because a constraint set changes the string and is therefore part of what "SELFIES" means in
these results. InChI needed nothing: zero failures on the same pool.

Dropping the un-encodable molecules was never available as an option — the ladder compares three
strings for the *same* molecules, so a row missing from one arm makes that arm a different sample
rather than a different notation. The machinery for it exists anyway (`UNENCODABLE`, and the probe
drops such rows from *every* arm and reports the count) because that failure is otherwise silent, but
on this pool it fires zero times.

**Launched 2026-09-06:** the `notation_selfies_s{0,1,2}` and `notation_inchi_s{0,1,2}` cells of
`configs/runs/molecule_generalist.jsonc`, at arm 2's recipe — the same file, so the recipe is shared
by construction rather than by copy. Four further decisions were forced by running it, and each is a
disclosure rather than a detail.

*Four Tier-A families stay in SMILES on every arm.* `ring_membership`, `aromatic_ring`, `ring_size`
and `fg_atom_membership` ask about a *named* atom, and only SMILES can mark one (`[cH:14]`). Holding
those four at SMILES on all three arms keeps the **mixture identical** — same families, same shares,
same rows — so a gradient between arms cannot be a difference in what they trained on. The cost is
that ~11 % of the mixture is the same string on every arm, diluting the contrast by that much
(`adapters/molecules.py::_notation_for`).

*Graph-to-SMILES becomes a translation rather than a canonicalization.* The SMILES arm's matched task
needs a randomised input or it is a copy; the canonical-only notations need none, since the target is
canonical SMILES and a canonical SELFIES input is already a translation. That makes g2s a *different
task* on those arms — one more reason §5's rule holds and the g2s column is never an arm comparison.

*`perm_spread` is dropped from these runs and `leakage` is kept*, which is the same distinction in two
directions. Re-ordering atoms is a property of the *string*, and a canonical-only notation has no
re-ordered form, so a permutation sweep would report a spread of zero — the tightest possible
Property-1 pass on an arm that never had the property. The validator refuses instead, and
`validators: "notation"` stops the refusal firing once a run. Stripping stereochemistry is a property
of the *molecule*, so every notation can write the stripped form and §3.2.10's leakage detector
survives the ladder intact (`evaluate/builtin.py::_write_notation`).

*Truncation is arm-asymmetric and is not fixed here.* Rows hitting `max_length` 512 on the Tier-B
prompts: SMILES 0–1.05 %, InChI 0–2.17 %, SELFIES 0–2.87 % (worst case SIDER; every other corpus under
0.3 %). Raising `max_length` on one arm puts a second uncontrolled axis under the ladder, and raising
it everywhere means rebuilding and re-running §8 — so 512 stays, matching arm 2 exactly. It biases
*against* the longer notations, the same direction as the effect being measured, which is the
uncomfortable direction rather than the safe one, and is why it is stated here. Evaluation excludes
truncated rows from every arm; training cannot.

#### 0.2 RESULT — 2026-09-07: the flat arm's advantage reverses with notation exposure

Six cells trained and annealed overnight (`002_*`, `003_*`), then all **twelve** annealed
checkpoints — the six new ones and arm 2's six — re-scored on one instrument by
`tools/notation_probe.py --checkpoint`, which excludes the truncated rows the default validators do
not. `tools/notation_ladder.py` assembles it. Mean over three seeds, ± seed sd:

| set | SMILES | InChI | SELFIES | graph |
|---|---:|---:|---:|---:|
| BACE | 0.8667 ±0.010 | 0.7888 ±0.021 | 0.8229 ±0.014 | 0.8185 ±0.019 |
| BBBP | 0.7086 ±0.014 | 0.6909 ±0.008 | 0.6940 ±0.011 | 0.7072 ±0.012 |
| HIV | 0.7291 ±0.010 | 0.7372 ±0.039 | 0.6896 ±0.019 | 0.7374 ±0.006 |
| Tox21 | 0.8100 ±0.016 | 0.7724 ±0.019 | 0.7677 ±0.007 | 0.7971 ±0.007 |
| SIDER | 0.8468 ±0.011 | 0.8351 ±0.001 | 0.8358 ±0.008 | 0.8332 ±0.008 |
| **five-set mean** | **0.7922** | **0.7649** | **0.7620** | **0.7787** |

**The gradient, which is what this tier exists to produce:**

| flat arm | − graph |
|---|---:|
| SMILES | **+0.0135** |
| InChI | **−0.0138** |
| SELFIES | **−0.0167** |

**The flat arm's advantage does not merely shrink as notation exposure falls — it reverses.** Written
in SMILES the flat arm beats the graph arm by +0.014; written in either notation the backbone has
read far less of, it *loses* to the graph arm by 0.014 to 0.017. That is a swing of about 0.03 across
the ladder, and it is the shape §8's pretraining reading predicts. On this evidence the flat arm's
edge in §8 is not a property of flat serialization — it is a property of *SMILES*, and it is
borrowed from pretraining rather than earned by the representation.

Two supporting details. The SMILES and graph legs re-score to 0.7922 and 0.7787 against §8's 0.7969
and 0.7795, so the corrected instrument barely moves the headline pair and §8's null stands as
written. And InChI and SELFIES land within 0.003 of each other despite differing in token length and
grammar — consistent with both sitting far below SMILES in exposure and the gap between *them* not
mattering much, which is what a saturating dose-response looks like.

**What this does not establish, stated plainly.**

* **Three seeds.** The per-dataset resolution is 0.077 and the pooled resolution ~0.021
  (§8.4.8). The SMILES-to-SELFIES swing of 0.030 clears the pooled line; the individual gaps
  (0.0135, 0.0167) do not clear it by much. This is a consistent direction across two independent
  notations and five sets, not a resolved effect size.
* **Truncation still biases this result in its own favour.** The longer notations lose more training
  rows to `max_length` (SMILES 0–1.05 %, InChI 0–2.17 %, SELFIES 0–2.87 %), and a lost row is one
  whose answer was cut off — so the notation arms trained on slightly more corrupted supervision.
  That handicap points the *same way* as the reversal. Evaluation excludes those rows on every arm,
  so the readout is clean; training cannot be, and the honest position is that some unknown part of
  the 0.03 swing is this artifact rather than the prior. Raising `max_length` for all four arms and
  re-running is what would settle it.
* **~11 % of the mixture is identical SMILES on every arm** (the four atom-level Tier-A families),
  which dilutes the contrast — the true notation effect is larger than what this ladder measures.
* **The exposure ordering rests on the corpus-frequency argument**, not on a measurement: Tier 0.1
  found every arm at chance zero-shot and produced no ranking to order the ladder by.

**Excluded rows, union across all four arms, dropped identically from each** (which is what
`check_arms_agree_on_labels` verifies): SIDER 297, Tox21 49, HIV 4, BBBP 1, BACE 0 — all truncation,
zero un-encodable.

**0.3 — The `bias: none` control at the tuned recipe.** ~6 GPU-h, `molecules/PLAN.md` §10.3.1, still
owed and independent of the notation question. Until it exists, "the graph arm" cannot be decomposed
into structural channel versus extra adapter capacity at the settled numbers, and every mechanism
sentence in §8 is really a statement about the whole arm. The specialist form is the committed item
because it is cheap and it closes the Tier-B decomposition. The stronger form — a `bias: none` graph
arm inside the generalist, three seeds, ~57 GPU-h, a config change and no code since `bias: "none"` is
already legal on the graph arm — decomposes §8's *probe* win instead, which is where the graph arm
actually leads. Run the stronger form only if 0.1 and 0.2 make the mechanism worth pinning down.

### Tier 1 — are the two channels complementary or redundant? (~60 GPU-h)

**The graph+SMILES arm.** Three seeds, one rebuild. A third `arm` value holding both
representations: the `rich_levi` graph exactly as the graph arm has it, plus the flat arm's SMILES
appended to the question node's text.

This answers a different question from Tier 0. The notation gradient asks where the flat arm's edge
comes from;
this asks whether the structural channel carries anything the string does not already deliver to a
model that reads strings well. If graph+SMILES beats both single arms, that is the positive result the
null currently denies the campaign. If it lands on the flat arm, the channel is redundant given SMILES
at 1B — a clean bound, and a more useful sentence than a tie.

Decisions to make before it runs, each with a reason:

* **The SMILES goes in the question node's text, not its own node.** A SMILES node would need an SPD
  relationship to the atoms that has no canonical answer, and appending makes the arm literally the
  union of the two existing ones, which is what keeps the three-way comparison readable.
* **Graph-to-SMILES becomes canonicalization**, exactly as in the flat arm (§5) — otherwise the prompt
  spells out the answer. Keep it in the mixture at its 0.15 share so the shares stay identical to
  arm 2 and only that task's difficulty moves, which §5 already discloses.
* **This arm forfeits Property 1**, and that is not a detail. With SMILES in the prompt the
  permutation invariance in §8 is gone. So it can never be the headline model of the thesis — it is a
  diagnostic that says what the structural channel adds, and the write-up has to label it that way or
  it undercuts the strongest molecule-specific claim the campaign has.
* `config.py`'s `ARMS` gains the value and the flat-arm bias assertion has to stop catching it; the
  arm is part of the source cache key, so it builds its own datasets.

**The `adapt` forks land here too, and are not in that 60** — eighteen short runs, three held-out
tasks × two starting points × three seeds (§4), each a few hundred steps on one card, so the total is
small but unmeasured until one is timed. Steps-to-target is the one held-out measurement arm 2
promised and did not make, and the Tier-1 arm is a third starting point worth including once it
exists.

### Tier 2 — remove the budget artifact, then extend the horizon — RUN 2026-09-09

**Result: the budget artifact was real and removing it did not settle the arm question.** Both arms
improved at twice the horizon — graph +0.0109, SMILES +0.0069 on the five-set mean — and the flat
arm's lead narrowed from +0.0135 to +0.0095. But the *paired* evidence got weaker rather than
stronger, and the finding is that three seeds no longer resolve this gap:

| | per-seed SMILES − graph | mean | t(2) | sign |
|---|---|---:|---:|---:|
| 1× (5,599 steps) | +0.0191, +0.0056, +0.0160 | **+0.0135** | +3.31 | **3/3** |
| 2× (11,140 steps) | +0.0296, +0.0095, −0.0106 | **+0.0095** | +0.82 | 2/3 |

The point estimate moved 0.0040 — nothing — while the sd of the paired difference **tripled**,
0.0071 → 0.0201. What was lost is resolution, not the gap. **HIV on the flat arm is the whole of the
new variance**: its per-seed scores go from [0.7370, 0.7319, 0.7185] at 1× to [0.8588, 0.7737,
0.7283] at 2×. HIV is exactly the corpus the fix moved most (1.04 → 2.21 epochs), so giving the flat
arm twice as much of it made its HIV score seed-dependent rather than uniformly better. Anything
built on this comparison needs more than three seeds.

Two per-task findings that do not depend on that mean, from `tools/horizon_compare.py`:

* ~~**The graph arm's g2s failure is structural, not under-training.**~~ **Withdrawn 2026-09-10 — it
  was an instrument defect.** The reading was that at twice the budget the flat arm goes 0.0193 →
  **0.0720** exact match and 0.1527 → **0.3127** validity while the graph arm stays at exactly 0.0000
  across 1,500 attempts, so doubling taught one arm to write molecules and the other nothing. What
  the horizon actually taught the flat arm was to *stop*: no generative answer in this build carried
  an end-of-text token (§8, "A stop-token defect found on 2026-09-10"), and the graph arm at 2× emits
  the exactly-correct canonical SMILES as a **prefix** of its output 46.5 % of the time before
  running on to the generation cap. It is the strongest per-task result the graph arm has, and it was
  reported as its worst.
* **The held-out topology margin did not widen, and one half of it shrank.** `bond_path` goes 1.6× →
  2.0× the best flat arm, `longest_chain` 2.2× → 1.5× (graph 0.1013 → **0.0687**). Both are noisy at
  n=3, so the honest statement is that the clearest evidence for the graph arm is *not* reinforced by
  a longer run.

The nine in-mixture structural probes all improve on both arms and the ranking is unchanged (graph
takes 7 of 11, against 8 of 11 at 1×). ChEBI-20 is flat to within noise on both arms, as at 1×.

`configs/probes/006_molecule_generalist_2x.jsonc` is the config; the settings and the resolved
distribution follow.

**§2's sampling fix is implemented** (`budget_scale`, `task_passes`), so this tier is now a run rather
than a design. One setting — `"budget_scale": 2.0` — gives **11,140 steps against arm 2's 5,599**,
636,476 examples, and this distribution:

| task | share | preset | epochs | |
|---|---:|---:|---:|---|
| `mol/chebi20` | 0.1930 | 0.2000 | 6.00 | capped |
| `mol/tox21` | 0.1564 | 0.1465 | **1.85** | was 0.86 |
| `mol/g2s` | 0.1513 | 0.1500 | — | generator |
| `mol/hiv` | 0.1134 | 0.1062 | **2.21** | was 1.04 |
| `mol/sider` | 0.1106 | 0.1036 | 4.57 | was 2.14 |
| Tier A, each of 9 | 0.0280 | 0.0278 | — | generator |
| `mol/bbbp` | **0.0117** | 0.0235 | 6.00 | capped — thinned, not exhausted |
| `mol/bace` | **0.0114** | 0.0203 | 6.00 | capped |

Blocks land at Tier B 40.3 %, Tier A 25.2 %, ChEBI 19.3 %, g2s 15.1 %. **Every corpus is at six
repeats or fewer** — nothing is seen more often than arm 2 saw it — and the only structural change is
that the three that run out are drawn less rather than allowed to end the run. **HIV and Tox21 cross
one epoch for the first time**, which is the whole point: they are the two corpora §8's property gap
is most plausibly an artifact of.

ChEBI-20's 0.7-point shortfall is the block-of-one case resolving itself. It could be held at exactly
20 % with `"task_passes": "mol/chebi20=7"`, and that is the wrong trade — it buys 0.7 points of share
by showing every caption a seventh time, and a 0.7-point drift is far less of a confound than one
source repeating more than the rest.

Two ceilings before reaching higher: the 80 % block floor lets ChEBI carry the mixture to 2.41×, and
BACE's 1 % floor binds first at **2.28×**. So ~2× is not a round number, it is close to what this
mixture actually supports.

Shrink the `val` role in the same rebuild (§3). Then one graph/flat pair at the longer horizon, one
seed first, to see whether the property numbers move at all.

**Cost:** the horizon is 1.99× arm 2's, and a graph/flat pair at arm 2's horizon measured 27 GPU-h, so
budget ~55 GPU-h for a pair plus its anneals. Confirm against the first cell's measured s/step rather
than trusting the multiplication — a walltime extrapolated across two factors at once has come in
1.9× low before.

This is the only "scale up" worth doing before a mechanism is in hand, and §8 gives a reason to expect
something: the anneal alone took the graph arm's worst probe deficit from 0.698/0.859 to 0.877/0.910.
It is also a prerequisite for Tier 3 rather than an alternative to it — the same rule binds harder at
every larger size, and `molecules/PLAN.md` §10.1's decision to hold the epoch budget fixed across the
ladder is only defensible once the budget rule itself is not the artifact.

### Tier 2b — the graph arm's graph-to-SMILES ceiling, on a fixed instrument (~26 GPU-h)

**Not run, and superseded before it started.** It was launched on base weights on 2026-09-10 and
cancelled the same day, because Tier 2c answers the question it was built to ask — "where does the
number land when the model is trained with a stop token" — at the campaign's own 15 % g2s share and on
the backbone the ladder will use, rather than at a specialist's 100 % share on a backbone being
retired. The ceiling question it *uniquely* asks (how much of the 0.4193 is share rather than
capability) stays open and is now cheap: the same two cells, on instruct weights, differenced against
Tier 2c's graph cell. The design below stands as written.

`configs/probes/007_g2s_specialist.jsonc`. Two cells, graph and SMILES, seed 0, `mol/g2s` and nothing
else, 11,140 steps — the 2× horizon, with the whole budget on one task: **625,299 g2s examples
against 96,300 at 2× and 47,700 at 1×**, about 14 draws over the 44,088-molecule train pool.

**Why it is worth 26 GPU-h now and was not worth it yesterday.** Under the stop-token defect the
question was "can the graph arm serialize a graph at all", and a specialist was the way to separate
"a 15 % share is not enough gradient" from "the arm cannot do it". The defect answers that: the arm
*can*, at a 46.5 % prefix rate, with 15 % of the mixture. What is not known is where the number lands
when the model is trained with a stop token, and no amount of re-scoring an existing checkpoint gets
at it — the prefix rate is a floor read off a model that was never taught to end a string.

The ceiling matters beyond g2s. Every other task in this campaign reads out at one position, so g2s
is the only place the structural channel has to survive a *sequence* of decisions. A high ceiling
makes graph-to-text generation a claim this architecture can carry; a low one localises the limit to
the writing position, where SPD from the prompt node is three-valued by construction — 0 to itself, 1
to every atom, 2 to every bond node, verified identical on 200 test molecules — so the query has no
ordering signal even though the keys do.

The SMILES cell is a recipe control, not an arm comparison (§5), and it is also the arm the defect
hurt least (33.5 % runaway against 98.7 %), which makes it the cheapest read on how much of any
movement is the fix rather than the budget.

Open, and not for this probe to answer: **§8's ChEBI-20 and g2s numbers were all measured through the
defect.** Re-measuring them means re-training the campaign's cells with `answer_eos` on, which is a
campaign-sized bill, not a probe — and that is the bill Tier 2c paid.

### Tier 2c — the switch to instruct weights and chat formatting — RUN 2026-09-10, READ OUT 2026-09-11

**This is the campaign the top of this document reports.**
`configs/probes/008_molecule_generalist_instruct.jsonc`. Six cells — graph and SMILES, three seeds —
at `006`'s horizon, mixture, budget rule and molecules, on `meta-llama/Llama-3.2-1B-Instruct` in chat
formatting. **The paradigm the trunk is going to live in, moved at the scale where it is cheap to
debug.**

`generalist/PLAN.md` D4 targets a 7–12B instruction-tuned backbone; D3 — decided — pairs instruct
weights with chat formatting, "both together, neither alone". Everything before this was base weights
in `Q:/A:` formatting, which was right while the question was what the prefix nodes contribute
(`molecules/PLAN.md` §8.4.4) and is the wrong place to still be standing when the ladder starts.

**What the chat format is, for a graph.** The question node carries the user turn and the prompt node
the assistant turn:

```
question node   <|start_header_id|>user<|end_header_id|>\n\nQuestion: …<|eot_id|>
prompt node     <|start_header_id|>assistant<|end_header_id|>\n\n CCO<|eot_id|>
```

That is the only faithful reading available: the graph arm's other nodes are a *set*, with no linear
order to place a turn marker into. The flat arm is one node and gets the whole template, with the
molecule inside the user turn — which keeps Property 2 exact, since a flat arm is still a single-node
graph whose biases are identically zero.

Three deliberate departures from `apply_chat_template`, each forced:

* **No `<|begin_of_text|>`.** There is no first node to put it on — `ordering: rcm` permutes them and
  Property 1 says the model must not care. Putting one on the flat arm alone would unmatch the arms.
* **No system turn.** The stock template injects a "Cutting Knowledge Date" block containing *today's
  date*, which would make a build's bytes depend on the day it ran.
* **The answer keeps its leading space** (`"\n\n Yes"`). That space is what makes the supervised token
  `" Yes"` — the same id the margin readout has always scored — so a chat-format number stays
  comparable to a plain-format one.

**The terminator is the format's, not the appender's.** `<|eot_id|>` closes the assistant turn in the
*text*, so it falls inside the supervised span and `answer_eos` does not also append one; a doubled
stop token would train through silently and show up only as a metric that would not move. And it is
applied to the **generative kinds only**: a `token` or `yesno` answer is read as a logit margin at the
prompt node's last token, so terminating those turns would score the terminator instead of the answer.
`test_chat_format_leaves_the_scored_position_alone` pins that.

**Cost.** Both arms on two cards (`accumulation_steps` halved, so the optimizer step is `006`'s and
only the wall clock moves), one 24 h job per cell instead of `006`'s four chained 12 h chunks. Putting
the *flat* arm on two cards as well is not waste — the graph arm is ~4× its cost per step, and
matching the card count is what makes six cells finish together instead of three finishing and three
running on alone.

**What is comparable, and what is not.** Same mixture, same seeds, same 11,140 steps, same molecules
(`_draw_rng` knows nothing about the backbone or the format), so the property-prediction rows are a
clean base-vs-instruct comparison. The **generation rows are not**: `006`'s g2s and ChEBI-20 numbers
were measured with no stop token at all, so that column scores whether the model stopped. It is a
floor, not a baseline. Read with `tools/horizon_compare.py --legs backbone`.

The chat format costs ~9 tokens an example on both arms, so at a pinned 11,140 steps the example
budget lands ~3 % under `006`'s. Steps are the invariant worth holding — the schedule is defined on
them — and the difference is disclosed rather than tuned away.

#### 2c RESULT — 2026-09-11: the graph arm writes molecules, and writes them better than the flat twin

All six cells trained to 11,140 steps and annealed to 12,255. Trunk cost: graph 8.4–10.2 h at
2.7–3.3 s/step on two cards, flat 2.0–2.5 h. Read with
`tools/horizon_compare.py --legs backbone`.

**Graph-to-SMILES, which was 0.0000 and is now the best number in the campaign.**

| | graph base | graph instr | SMILES base | SMILES instr |
|---|---:|---:|---:|---:|
| `exact_match` | 0.0000 | **0.4193** ±0.0291 | 0.0720 | 0.2113 ±0.0031 |
| `roundtrip_match` | 0.0000 | **0.4613** ±0.0261 | 0.0953 | 0.2607 ±0.0101 |
| `validity` | 0.0480 | **0.7560** ±0.0156 | 0.3127 | 0.6607 ±0.0117 |

The 46.5 % prefix rate §8 measured was a floor on a real capability, and the fixed instrument lands
close to it: **0.4193 exact match**. The graph arm's mean prediction is now 45.8 characters against a
44.8-character target — it has learned where a molecule ends.

By heavy-atom count (seed 0, whole 1,000-molecule test split, `tools/g2s_report.py`):

| heavy atoms | n | graph valid | graph exact | SMILES valid | SMILES exact |
|---|--:|--:|--:|--:|--:|
| 0–10 | 39 | 0.8462 | **0.5385** | 0.8718 | 0.4359 |
| 11–15 | 117 | 0.7949 | **0.4957** | 0.7692 | 0.2821 |
| 16–20 | 231 | 0.8095 | **0.5238** | 0.7316 | 0.2814 |
| 21–30 | 400 | 0.7850 | **0.4875** | 0.6400 | 0.1975 |
| 31+ | 213 | 0.5681 | **0.2160** | 0.4554 | 0.0892 |

The graph arm is ahead in every bucket, and **flat across 10–30 heavy atoms** rather than decaying
with size — it degrades only past 31 atoms, where the node budget also bites. §5's rule still holds
and these columns are still not an arm comparison (the flat twin is doing canonicalization from a
string that already spells the answer). What the graph column says on its own terms is that
serializing a graph is something this architecture does, at 1B, on 44 % of molecules exactly.

**Attribution, stated honestly: two things changed at once.** The stop-token fix and the
instruct+chat switch landed in the same build, so the generation deltas above cannot be split between
them. The base column is a floor, not a baseline. Separating them needs a base-weights run with
`answer_eos` on, which nothing yet requires.

**ChEBI-20 captioning roughly doubled on both arms** — BLEU-2 0.194 → 0.426 (graph) and 0.198 → 0.457
(flat), ROUGE-L 0.306 → 0.505 and 0.312 → 0.533. Same confound, same reading.

**Property prediction: the gap closed, and the sign flipped.**

| leg | graph | SMILES | SMILES − graph | sd | t(2) |
|---|--:|--:|--:|--:|--:|
| base 2× | 0.7896 | 0.7991 | +0.0095 | 0.0201 | 0.82 |
| instruct | **0.7988** | 0.7966 | **−0.0022** | 0.0126 | −0.31 |

Neither is significant (t critical at df=2 is 4.303), and the honest statement is the one §9 Tier 2
already forced: this comparison is noise-dominated at three seeds. But the flat arm's lead is gone,
the graph arm is now nominally ahead, and the seed spread *narrowed* (0.0201 → 0.0126) rather than
growing. Tox21 carries most of it (graph +0.0374, the largest single move); SIDER +0.0092.

**The structural probes did not move**, which is the control this needs: the nine in-mixture
exact-match tasks are within ±0.02 on both arms, and the held-out topology pair is unchanged (graph
`bond_path` 0.0780 → 0.0760, `longest_chain` 0.0687 → 0.0780). The backbone swap did not quietly
change what the structural channel contributes; it changed what the model can *say*.

**Property 1 holds 15/15**, and holds at the floor. Every (set, seed) cell has
`perm_spread/.../within_tolerance` at 1.0 with `margin_spread_max` **equal to or below its own
`margin_control_max`** — the spread over ten relabelings of a molecule is not larger than the spread
the same run measures over ten *re-runs of the identical input*. In grid terms that is 2 to 4 quanta of
the margin, which is the resolution of the instrument rather than a property of the model. The chat
wrapper did not perturb it, which is the thing to check when the prompt node's text changes.

##### All four cells, task by task

Mean ± sd over seeds 0/1/2. **`gap` is graph − SMILES**, so positive is the graph arm ahead. The
`base` columns are `006` (base weights, `Q:/A:`); the `instr` columns are this tier. Reproduce with
`tools/horizon_compare.py --legs backbone`.

**Property prediction — ROC-AUC**, from `tools/notation_probe.py --checkpoint`, which drops the union
of truncated rows across arms so every cell scores the identical 500 molecules.

| task | base graph | base SMILES | gap | instr graph | instr SMILES | gap |
|---|---|---|--:|---|---|--:|
| BACE | 0.8263 ±0.0513 | 0.8670 ±0.0092 | −0.0407 | 0.8303 ±0.0390 | 0.8599 ±0.0208 | −0.0296 |
| BBBP | 0.7137 ±0.0125 | 0.6883 ±0.0172 | +0.0254 | 0.7061 ±0.0237 | 0.6973 ±0.0333 | +0.0088 |
| HIV | 0.7587 ±0.0317 | 0.7869 ±0.0663 | −0.0282 | 0.7618 ±0.0373 | 0.7575 ±0.0449 | +0.0042 |
| SIDER | 0.8495 ±0.0030 | 0.8424 ±0.0061 | +0.0070 | 0.8586 ±0.0044 | 0.8449 ±0.0037 | +0.0138 |
| Tox21 | 0.7999 ±0.0162 | 0.8109 ±0.0015 | −0.0110 | 0.8373 ±0.0136 | 0.8231 ±0.0078 | +0.0142 |
| **five-set mean** | **0.7896** | **0.7991** | **−0.0095** | **0.7988** | **0.7966** | **+0.0023** |

The graph arm goes from behind on 3 of 5 to ahead on 4 of 5, and **BACE is the only set where SMILES
clearly leads** in either leg. HIV is the set that moves most across the swap (−0.0282 → +0.0042) and
it is also the one Tier 2 identified as the whole of the new seed variance, so read it as the least
settled row here rather than as the mechanism.

**Structural probes — exact match.** The first nine are in-mixture; the last two are held out and no
training source touches them.

| task | base graph | base SMILES | gap | instr graph | instr SMILES | gap |
|---|---|---|--:|---|---|--:|
| `aromatic_ring` | 1.0000 ±0.0000 | 1.0000 ±0.0000 | +0.0000 | 0.9967 ±0.0023 | 0.9993 ±0.0012 | −0.0027 |
| `ring_membership` | 1.0000 ±0.0000 | 0.9907 ±0.0031 | +0.0093 | 1.0000 ±0.0000 | 0.9940 ±0.0035 | +0.0060 |
| `ring_size` | 0.9627 ±0.0110 | 0.8907 ±0.0190 | **+0.0720** | 0.9667 ±0.0122 | 0.8893 ±0.0266 | **+0.0773** |
| `ring_count` | 0.9040 ±0.0080 | 0.9353 ±0.0081 | −0.0313 | 0.8993 ±0.0101 | 0.9380 ±0.0122 | −0.0387 |
| `fg_presence` | 0.9953 ±0.0012 | 0.9827 ±0.0061 | +0.0127 | 0.9933 ±0.0046 | 0.9873 ±0.0012 | +0.0060 |
| `fg_count` | 0.9667 ±0.0050 | 0.9760 ±0.0035 | −0.0093 | 0.9640 ±0.0020 | 0.9700 ±0.0087 | −0.0060 |
| `fg_atom_membership` | 0.9953 ±0.0050 | 0.9593 ±0.0064 | +0.0360 | 0.9973 ±0.0031 | 0.9760 ±0.0122 | +0.0213 |
| `stereo_assigned` | 0.9847 ±0.0076 | 0.9900 ±0.0020 | −0.0053 | 0.9860 ±0.0040 | 0.9973 ±0.0012 | −0.0113 |
| `stereo_potential` | 0.8573 ±0.0050 | 0.8293 ±0.0162 | +0.0280 | 0.8347 ±0.0140 | 0.8433 ±0.0170 | −0.0087 |
| `bond_path` *(held out)* | 0.0780 ±0.0320 | 0.0387 ±0.0099 | +0.0393 | 0.0760 ±0.0040 | 0.0473 ±0.0023 | +0.0287 |
| `longest_chain` *(held out)* | 0.0687 ±0.0117 | 0.0453 ±0.0101 | +0.0233 | 0.0780 ±0.0035 | 0.0240 ±0.0151 | +0.0540 |

**Every gap keeps its sign across the backbone swap except `stereo_potential`**, which is the weakest
of them and crosses zero at 0.0087 against a seed sd of 0.017. That stability is what this table is
for: `ring_size` (+0.072 → +0.077), `ring_count` (−0.031 → −0.039) and the held-out pair are properties
of the *channel*, not of the formatting. On the held-out tasks the graph arm now leads `longest_chain`
by 3.3× and `bond_path` by 1.6×, both at floor-level absolute accuracy — the ordering is the finding,
not the score.

**Generation.** The `base` column here was measured with no stop token (§8): it scores whether the
model stopped and is a floor, not a baseline. Only the two `instr` columns are a comparison.

| task | base graph | base SMILES | gap | instr graph | instr SMILES | gap |
|---|---|---|--:|---|---|--:|
| `g2s/exact_match` | 0.0000 ±0.0000 | 0.0720 ±0.0072 | −0.0720 | 0.4193 ±0.0291 | 0.2113 ±0.0031 | **+0.2080** |
| `g2s/roundtrip_match` | 0.0000 ±0.0000 | 0.0953 ±0.0042 | −0.0953 | 0.4613 ±0.0261 | 0.2607 ±0.0101 | **+0.2007** |
| `g2s/validity` | 0.0480 ±0.0174 | 0.3127 ±0.0291 | −0.2647 | 0.7560 ±0.0156 | 0.6607 ±0.0117 | **+0.0953** |
| `chebi20/bleu2` | 0.1937 ±0.0100 | 0.1975 ±0.0083 | −0.0039 | 0.4257 ±0.0021 | 0.4567 ±0.0095 | −0.0309 |
| `chebi20/bleu4` | 0.1354 ±0.0121 | 0.1383 ±0.0100 | −0.0029 | 0.3078 ±0.0025 | 0.3396 ±0.0096 | −0.0318 |
| `chebi20/rouge_l` | 0.3064 ±0.0116 | 0.3118 ±0.0078 | −0.0054 | 0.5050 ±0.0018 | 0.5334 ±0.0077 | −0.0285 |
| `chebi20/meteor` | 0.4704 ±0.0262 | 0.4768 ±0.0222 | −0.0063 | 0.4941 ±0.0016 | 0.5253 ±0.0065 | −0.0312 |

ChEBI-20 read as a tie under the defect and does not now: as both arms roughly doubled, the flat arm's
lead grew 5–11× depending on the metric, to 4–7 seed-sds. Captioning is the one family where the flat
arm holds a consistent edge at the resolution available, and it is the expected direction — a caption
is text about a molecule, and the flat arm's molecule is already text.

### Tier 3 — the backbone ladder, if the mechanism earns it (~170 GPU-h at 3B)

`molecules/PLAN.md` §10.1 verbatim, but **3B only and not 8B**, and only after Tiers 0–2 have either
produced a mechanism worth carrying up a ladder or sourced §10.1's recollection with a measurement.
`lr` is re-screened at the new scale rather than inherited — §10.1 is emphatic about this and it is
the campaign's most expensive recurring mistake. Six cells at ×2.1 is ~170 GPU-h plus the screen.

**The rung is `Llama-3.2-3B-Instruct`, not `Llama-3.2-3B`.** Tier 2c settled the backbone question
for the whole ladder: every rung is an instruction-tuned model in chat formatting, so that what the
ladder varies is size and nothing else. `resolve_prompt_style` picks the format off the model name, so
this costs a config line and no code.

### What every tier carries

**Pool over the five property sets.** Three seeds resolve only 0.077 on one dataset; pooled the
resolution is ~0.021, and every effect Tier 0 and Tier 1 are chasing — the flat arm's 0.0174 lead
included — lives below the per-dataset line and above the pooled one. Decide the seed count from the
effect actually expected, at the scale it is expected at, which is the discipline §8.4.8 says this
campaign got wrong once already.

**The general-text held-out loss**, adapter-on against adapter-off. Still not measured, still owed,
and cheap: the assistant goal makes text ability something these campaigns should report, and no
validator measures it.

**Bucket `B` for training batches** before anything runs distributed at width (`DESIGN.md` §D9). Until
that lands, price graph cells at one card — four ranks made the arm-2 anneal slower than one would
have.
