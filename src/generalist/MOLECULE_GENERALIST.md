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

- [ ] **Gate 1.** The tuned three-seed re-run has reported: the specialist (arm 1) numbers on
      BACE / BBBP / HIV at the settled `lr` and `lora_r 16`, both arms. Arm 2 is compared against
      *these*, not against the `lr 3e-5` tables.
- [ ] **Gate 2.** `DESIGN.md` §D1–D6 built and the plumbing smoke run has passed (schema
      round-trip, partition disjointness, per-task gradient share == mixture weight, resume
      bit-exact).
- [ ] **Build.** Molecules adapter, the partition, graph-to-SMILES generator, ChEBI-20 loader,
      the isomeric round-trip test.
- [ ] **Cross-check.** One specialist cell (BACE, both arms, one seed) trained *through the
      generalist harness* as a single-task mixture reproduces the molecules-trainer number within
      seed noise. Until this passes, arm 2 minus arm 1 is a trainer difference, not transfer.
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
leakage detector: at chance with `stereo_tags: off`, high with `on`. On the `014` pool it is
single-answer (§3.2.10.1) and would be void; the generalist's pool is the train-role union (§3),
which has to be measured for this family before the mixture is frozen.

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

**Passes.** Finite sources (Tier B, Tier C) get at most three passes. Generators (Tier A,
graph-to-SMILES) draw fresh examples every pass from the train-role pool, single-pass, so the
early-peak overfitting the specialist runs show cannot come from repetition. The total budget is
therefore *defined by the finite sources*: three passes of Tier B + C at their combined 0.60 share
fixes the number of examples, and the registry computes and records the step count from that
rather than taking it as a free knob.

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

## 4. Held out

| Held out | Why this one | Scored as |
|---|---|---|
| `bond_path` | Declared 2026-08-28 (`molecules/PLAN.md` §4.1). SPD *is* the answer by construction, and SPD is the graph arm's bias, so it is the cleanest test of whether the structural channel crosses question templates. | zero-shot exact match, then steps-to-target from the generalist vs from base Llama |
| `longest_chain` | Added 2026-09-02. With `bond_path` it makes the held-out set *the traversal family* while training covers rings, functional groups and stereo. Transfer from local motifs to path questions is a real claim; `ring_count` was the alternative and is weaker because `ring_size` and `ring_membership` are in training, so transfer there is near-duplicate. It remains measured as a specialist (`014`: graph 0.988 vs flat 0.828), so nothing is lost by holding it out. | same two ways |
| ClinTox | Declared 2026-08-28. A toxicity / trial-failure endpoint, unlike binding or permeability. | zero-shot AUROC, then steps-to-target |

Two Tier-A holdouts is the number. A third starts costing training coverage for a declaration
made after seeing results, which is worth less than the two made before.

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
`bias_lr 1e-2` on the graph arm, `weight_decay 0.1`, `max_spd 32` pending the clamp sweep. The
learning rate is whatever the three-seed re-run settles; `3e-4` at r16 is the provisional value.
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
