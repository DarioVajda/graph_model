# The molecule generalist — one model over every molecule task

**Status (2026-09-15):** the campaign is run and read out; what follows it is planned in §9. **The
reportable campaign is `configs/probes/008_molecule_generalist_instruct.jsonc`** — six annealed 1B
checkpoints, three seeds × two arms (graph, and a SMILES flat twin), on `Llama-3.2-1B-Instruct` in
chat formatting at the 2× horizon. That backbone is the one the ladder climbs (`generalist/PLAN.md`
D3/D4), so the headline numbers come from it.

Three earlier campaigns fed into it and are kept only for what they established (§8): arm 2 on base
weights at 1×, the notation ladder, and the doubled horizon. Their **property** rows stand. Their
**generation** rows do not — every one was measured through the stop-token defect (§8.5) and scores
whether the model stopped rather than whether it was right.

This document is the *what and why*; `DESIGN.md` is the *how*, and its §D9 carries the harness
contracts this campaign settled. For the notation ladder's full tables, one per family on base
weights, read `configs/runs/molecule_generalist.md`.

**Where it sits.** Arm 2 of the molecules plan (§4, "chemistry generalist") and the first consumer of
the generalist harness. Arm 1 is one specialist per task, closed at `molecules/PLAN.md` §8.4.8. Arm 3
— molecules folded into the cross-domain mixture — is the trunk's admission gate and is not run
separately.

**What comes next, in one sentence.** The annealed graph checkpoints are a *trunk*: §9 measures what
they are worth as a starting point — for a specialist on a task they trained on, for a task they never
saw, and for a small, curated set of free-form questions that turns the model into something a person
can ask.

---

## Summary

One 1B instruction-tuned model, both arms, trained on every molecule task the repo can produce — RDKit
structural questions, MoleculeNet property prediction, ChEBI-20 captioning and graph-to-SMILES — with
one molecule-level partition across all sources and three held-out tasks no training source touches.

**The graph arm writes molecules, and writes them better than the flat twin.** Asked for the SMILES of a
molecule it has only ever seen as a graph, it is exactly right **41.9 %** of the time (validity 0.756,
round-trip 0.461) against the flat arm's 21.1 % — and the flat arm's task is canonicalizing a string
that already spells the answer. By heavy-atom count the graph arm is ahead in every bucket and flat from
10 to 30 heavy atoms. This is the one place the structural channel has to survive a *sequence* of
decisions rather than read out at one position, and it survives.

**The property-prediction claim is a null, and the flat arm's lead is gone.** Five-set mean ROC-AUC:
graph **0.7988**, SMILES **0.7966**; paired difference −0.0022, sd 0.0126, t(2) = −0.31. Three seeds do
not resolve a gap this size. What changed against base weights is the sign and the spread: the flat
arm's +0.0095 lead became a −0.0022 deficit and the seed spread narrowed. The graph arm leads 4 of 5
sets; BACE is the one where SMILES clearly leads.

**Property 1 is the strong one, and the backbone swap did not touch it.** All fifteen graph cells pass
with their permutation spread at or below the *control* — two to four quanta of the margin, the floor
the instrument resolves. On base weights the flat arm's spread ran 10× to 40× wider and widened through
the anneal. No amount of pretraining beats a property.

**The structural split is stable across the backbone swap, which is what makes it a finding.** The
graph arm takes `ring_size` by +0.077, `fg_atom_membership` by +0.021, `ring_membership`, and both
held-out topology tasks — `longest_chain` by 3.3× and `bond_path` by 1.6× — while the flat arm keeps
`ring_count` and the stereo pair. Every gap holds its sign from base weights to instruct. The backbone
changed what the model can *say*; it did not change what the structural channel contributes.

**And a cost, on the axis never measured until now.** Asked 48 general-knowledge questions with no
molecule in them, the graph arm answers **one in seven with a ChEBI-20 caption** (0.139 ±0.012) against
the flat arm's 0.021 and the backbone's 0.000. Length, stop rate and single-token rate all score those
answers as healthy. The register survived; the topic did not. §7.6 has the mechanism and what it
implies for the trunk's forgetting control, and §9 acts on it.

**One earlier result bounds the second.** On base weights, writing the molecule as InChI or SELFIES
instead of SMILES turns a **+0.0135** flat-minus-graph property gap into **−0.0138** and **−0.0167**, at
the same molecules and mixture (§8.3). The flat arm's historical property edge was *SMILES*-specific —
an artifact of what the backbone was steeped in — which is the mechanism the closed gap is consistent
with. Three seeds, and a truncation artifact pointing the same way; the caveats are not small.

---

## 1. What goes in

Both arms train on everything below. The graph arm is `rich_levi`, `stereo_tags: on`, no SMILES
anywhere in the prompt — that is what makes graph-to-SMILES a real task rather than a copy. The flat
twin sees SMILES and gets the matched form of each task. Every source is routed by the question text
alone (the question node is `on`, D3).

| Source | Tier | Task form | Size | Role | Metric |
|---|---|---|---|---|---|
| `ring_membership`, `aromatic_ring`, `ring_size`, `ring_count` | A | 1–3 token exact answer | generator, capped per pass | train + in-mixture test | exact match |
| `fg_presence`, `fg_count`, `fg_atom_membership` | A | 1–3 token exact answer | generator, capped | train + in-mixture test | exact match |
| `stereo_potential`, `stereo_assigned` | A | 1–3 token exact answer | generator, capped | train + in-mixture test | exact match |
| BACE, BBBP, HIV | B | yes / no | 1.5k / 2.0k / 41k molecules | train + **headline** test | ROC-AUC from the yes/no margin |
| Tox21, SIDER | B | yes / no per endpoint | 78k / 39k (molecule, endpoint) pairs | train + diagnostic test | ROC-AUC per endpoint |
| ChEBI-20 | C | free-text caption | 26.4k / 3.3k / 3.3k | train + in-mixture test | BLEU-2/4, ROUGE-L, METEOR |
| graph-to-SMILES | — | canonical SMILES, stereo-free (§5) | generator, every train-role molecule once per pass | train + in-mixture test | validity, round-trip, canonical exact |

**Tox21 and SIDER are training signal, not results.** They have no anchor ladder (`molecules/PLAN.md`
§1), which is a problem for interpreting a number against the field and none for gradient. Together
they are the bulk of the property labels and take the mixture from three endpoints to about forty.
Their test AUROC is an internal diagnostic and never goes in the anchor table. Tox21's ~16k absent
labels are skipped at the (molecule, endpoint) level and the per-endpoint counts go in the run record.

**`stereo_assigned` is the suite's leakage detector** — at chance with the parity channel closed, high
with it open. On the generalist's train-role pool it is live (ten distinct answers) but skewed: the test
split answers 0 on 927 of 1,000, so the floor is 0.927 and only ~36 rows move when the channel closes.
It runs as the `leakage` validator, which closes the channel at *evaluation* with two scoring passes
and reports `void` on a single-answer split rather than a pass. The verdict is a floor test, nothing
finer.

**Excluded.** ESOL, FreeSolv and Lipophilicity as *tasks* — the margin readout cannot score a number
and none has an anchor — though their molecules serve as unlabeled generator pool. QM9, peptides and
text-to-molecule stay out for `molecules/PLAN.md` §1's reasons. A graph arm that *also* sees SMILES is
out of the campaign: it would turn graph-to-SMILES into a copy task and forfeit Property 1 (§8.7).

## 2. Mixture and budget

Weights are in *examples*, and by D7a each task's gradient share equals its example share (two-level
normalization, per-example within a task). **Loss is per-example everywhere**: captions are 50–100
tokens beside one-token answers, and token-summed loss would make Tier C most of the gradient at a
fifth of the examples.

| Block | design share | within the block |
|---|---:|---|
| Tier B (5 sets) | 0.40 | temperature ∝ size^0.5 — roughly BACE 5 %, BBBP 6 %, HIV 27 %, Tox21 37 %, SIDER 26 % of the block |
| Tier A (9 families) | 0.25 | uniform over families |
| Tier C (ChEBI-20) | 0.20 | — |
| graph-to-SMILES | 0.15 | — |

**Passes and horizon.** Finite sources (Tier B, Tier C) get at most **six** passes. Generators draw
fresh examples every pass from the train-role pool, so early-peak overfitting cannot come from
repetition. The budget is `min over finite corpora of (passes × train_size) / share`; within Tier B the
weight goes as `size ** 0.5` while availability goes as `size`, so **the smallest corpus always sets the
horizon**. At 1× that was BBBP at 5,599 steps, with HIV and Tox21 under one epoch — a live alternative
explanation for any property gap.

**`budget_scale` is the fix.** `registry.resolve` takes the budget as a multiple of that rule's own
feasible budget and **down-weights any corpus that cannot sustain its share** rather than letting it end
the run. Redistribution stays inside a block; a block of one (ChEBI-20) may shrink and spill, but no
block falls below 80 % of its design share; past that it is a refusal naming the block, and the way
through is an explicit `task_passes` override in the config and the hash. At 1.0 it is a no-op and the
1× hashes are unchanged. BACE's 1 % floor binds at **2.28×**, so ~2× is close to what this mixture
supports.

**The campaign runs at `budget_scale: 2.0`** — **11,140 steps**, 636,476 examples:

| task | share | design | epochs | |
|---|---:|---:|---:|---|
| `mol/chebi20` | 0.1930 | 0.2000 | 6.00 | capped |
| `mol/tox21` | 0.1564 | 0.1465 | **1.85** | was 0.86 at 1× |
| `mol/g2s` | 0.1513 | 0.1500 | — | generator |
| `mol/hiv` | 0.1134 | 0.1062 | **2.21** | was 1.04 |
| `mol/sider` | 0.1106 | 0.1036 | 4.57 | was 2.14 |
| Tier A, each of 9 | 0.0280 | 0.0278 | — | generator |
| `mol/bbbp` | **0.0117** | 0.0235 | 6.00 | capped — thinned, not exhausted |
| `mol/bace` | **0.0114** | 0.0203 | 6.00 | capped |

Blocks land at Tier B 40.3 %, Tier A 25.2 %, ChEBI 19.3 %, g2s 15.1 %. Nothing is seen more often than
six times, and HIV and Tox21 cross one epoch. ChEBI's 0.7-point shortfall is the block-of-one case
resolving itself; holding it at 20 % would cost a seventh pass over every caption, the worse trade.

## 3. The partition — one molecule, one role

The Tier-A generators and graph-to-SMILES draw molecules from the Tier-B corpora, so without one rule
across sources a structural question about a BBBP *test* molecule lands in training and the scaffold
split stops meaning "structurally novel" (`molecules/PLAN.md` §3.2.10 is the incident).

* **Key:** *stereo-free* canonical SMILES, computed once at adapter build time. Two stereoisomers have
  identical graphs up to the parity words, so keying on the isomeric string would let near-identical
  graphs straddle the line. Both isomers share one role; each keeps its own labels.
* **Roles:** `train`, `val`, `test`, `held_out`. Every molecule in every source gets exactly one.
* **Rule 1.** A molecule in any Tier-B or ChEBI-20 val/test split, or anywhere in ClinTox, is removed
  from *every* training source. Priority on conflict: `held_out > test > val > train`.
* **Rule 2.** Generators draw training molecules only from the `train` role.
* **Rule 3.** Generator *test* sets draw from `test`-role molecules, scaffold-novel by construction.
* **Rule 4.** The registry refuses to build a mixture violating rules 1–3, and the run record carries
  the per-role counts and the number of cross-source overlaps removed.

Enforced in `adapters/molecules.py`, pinned by a test that rebuilds the partition from the raw CSVs and
asserts pairwise disjointness (`DESIGN.md` §T2). **Every new source in §9 — replay prompts, curated
questions — takes its molecules through this partition and never from `test` or `held_out`.**

**The `val` role is larger than it needs to be** (7,690 molecules removed from every training source to
buy a diagnostic that never selects anything — WSD has no dev-score selection). A few hundred rows per
source would read the same curves. Shrinking it moves `build_version`, so it folds into the next
rebuild rather than happening on its own.

## 4. Held out

| Held out | Why this one | Scored as |
|---|---|---|
| `bond_path` | Declared 2026-08-28. SPD *is* the answer by construction, and SPD is the graph arm's bias, so it is the cleanest test of whether the structural channel crosses question templates. | zero-shot exact match, then steps-to-target from the trunk vs from base (§9.3) |
| `longest_chain` | Added 2026-09-02. With `bond_path` the held-out set is *the traversal family* while training covers rings, functional groups and stereo. Still measured as a specialist (`014`: graph 0.988 vs flat 0.828), so nothing is lost. | same two ways |
| ClinTox | Declared 2026-08-28. A toxicity / trial-failure endpoint, unlike binding or permeability. | zero-shot AUROC, then steps-to-target |

Two Tier-A holdouts is the number; a third costs training coverage for a declaration made after seeing
results. Zero-shot is measured (§7.3); **steps-to-target is not**, and it is §9.3.

Few-shot here means *few-example fine-tuning*, not in-context examples: several molecule graphs in one
prompt is not what the prefix-node layout is built for. The molecules package refuses to build these
three without `held_out_eval`, the registry mirrors the declaration, and a mixture naming any of them
fails in both places. **§9's curated question set must not contain a traversal or a trial-outcome
question either**, or the held-out set is spent by the back door.

## 5. Graph-to-SMILES and ChEBI-20

**Graph-to-SMILES** is the one task not in the molecules plan: the graph is on the *input* side, the
inverse of the text-to-molecule generation `molecules/PLAN.md` §1 excludes, and the bridge to
captioning.

* **The target is RDKit canonical SMILES, stereo-free**, `Chem.MolToSmiles(mol, isomericSmiles=False)`,
  for both arms. A parity word is only meaningful relative to a neighbour *ordering* and the graph has
  none, so a graph arm asked for `@`/`@@` would be asked for information it does not have. The flat
  twin's input carries stereo it must learn to drop; the graph arm's carries parity words it must learn
  to ignore. Emitting a stereo mark is an error and is recorded as a diagnostic.
* Three metrics in order of what matters: validity, round-trip match, canonical exact match.
* **The flat twin's matched task is canonicalization** — randomized SMILES in, canonical out. The graph
  arm has no input order to randomize and faces the hard version by construction. **The two columns
  measure different tasks and are never quoted as an arm comparison.**
* A generator with free labels, capped at one example per train-role molecule per pass.

**ChEBI-20** keeps its own split (26,407 / 3,301 / 3,300) and folds into the §3 partition; overlap with
MoleculeNet is measured, not assumed. It includes salts and multi-fragment molecules, so disconnected
graphs put SPD at the `max_spd` clamp between components; the heavy-atom cap is chosen against the
ChEBI size distribution and recorded. BLEU and ROUGE reward template matching, so a strong caption
number is weak evidence — this caveat travels with every Tier-C number.

## 6. Recipe, measurement, and what gets reported

**Recipe.** Both arms at `lora_r 16`, `lora_dropout 0.05`, `bias_lr 1e-2` on the graph arm,
`weight_decay 0.1`, `max_spd 32`. **`lr 1e-4`, matched across arms**, the lower of the specialist's two
per-task values, for two reasons: WSD holds the stable phase at `lr` for the whole run where the
specialists' cosine only touched its peak, so the same number is a larger dose here; and the risk is
asymmetric — 3e-4 cost the graph arm 0.109 ROC-AUC on HIV, which is 11 % of this mixture, while 1e-4 on
BACE and BBBP came out worse rather than broken. `lr_min 1e-5`.

**Schedule:** WSD — short warmup, constant stable phase for the §2 budget, one anneal fork decaying to
`lr/10` over ~10 % of the stable steps (1,115 steps on the 11,140-step trunk). **The annealed checkpoint
is the reportable model.** No test-set selection and no best-val selection: Tier-B validation
anti-ranks the arms on BBBP, and the anneal makes selection unnecessary. The trunk's own milestone
scores are recorded but are not results.

**The arms are matched in examples, not tokens.** The graph mixture measures 288 tokens an example
against the flat arm's 83, so one shared token budget would hand the flat arm 3.5× the batch. Graph
runs at `tokens_per_step 16384`, flat at **4689**, both landing on ~57 examples/step
(`tools/tokens_per_step.py`); `max_steps` pins the step count. Any new arm re-derives its own value.

**Backbone and format.** `Llama-3.2-1B-Instruct` in chat formatting — the paradigm the trunk lives in
(D3: instruct weights and chat template, both or neither), moved at the scale where it is cheap to
debug. The question node carries the user turn and the prompt node the assistant turn; that is the only
faithful reading for a graph, whose other nodes are a *set* with no linear order to put a turn marker
into. The flat arm is one node and gets the whole template. Three departures from the stock template,
each forced: no `<|begin_of_text|>` (there is no first node), no system turn (the stock one carries
*today's date*), and the answer keeps its leading space so the supervised token is the same ` Yes` the
margin readout has always scored. The terminator is the format's own `<|eot_id|>` on the generative
kinds only; a `token` or `yesno` answer is read as a logit margin and terminating it would score the
terminator. The chat format costs ~9 tokens an example, so at pinned steps the example budget lands ~3 %
under the base-weights run; disclosed, not tuned away.

**Two claims, never conflated** (`molecules/PLAN.md` §8.4.3.1): mean ± s.e. over seeds on the fixed
test set, which is what the anchors publish; and the paired per-molecule bootstrap, which is the
generalisation claim. HIV's effective n is ~132 actives, not 4,112 molecules.

**Disclosures that travel with every number:** Tox21 / SIDER are not anchor-comparable; the Tier-C
caveat; the flat twin's graph-to-SMILES is canonicalization; the partition counts; and since
2026-09-12, `caption_rate` beside every loss (§7.6).

**The cells.** Six cells of `008`, launched through `tools/chain.sh <file> <cell>`. The recipe is
written once at the top of the file and the cells differ only in arm and seed;
`test_the_campaign_cells_differ_only_where_they_are_meant_to` asserts it. All six read one build.
Both arms on two cards, one 24 h job a cell; graph 8.4–10.2 h of trunk, flat 2.0–2.5 h, ~80 GPU-h for
the campaign with anneals. Evaluation is batched and sharded (`DESIGN.md` §D9): a milestone costs ~12
minutes on one card, so evaluation cost is no longer a reason to thin what gets scored. Dataset builds
are CPU-only and are the long pole — 73 minutes for the full mixture; point `data_prep` at the arm's
own config or it over-builds.

## 7. Results — the instruct campaign

All six cells trained to 11,140 steps and annealed to 12,255. Mean ± sd over seeds 0/1/2; `gap` is
graph − SMILES, positive means the graph arm ahead. The `base` columns are the base-weights run at the
same horizon and molecules (`006`); reproduce with `tools/horizon_compare.py --legs backbone`.

### 7.1 Graph-to-SMILES

| | graph base | graph instr | SMILES base | SMILES instr |
|---|---:|---:|---:|---:|
| `exact_match` | 0.0000 | **0.4193** ±0.0291 | 0.0720 | 0.2113 ±0.0031 |
| `roundtrip_match` | 0.0000 | **0.4613** ±0.0261 | 0.0953 | 0.2607 ±0.0101 |
| `validity` | 0.0480 | **0.7560** ±0.0156 | 0.3127 | 0.6607 ±0.0117 |

The graph arm's mean prediction is 45.8 characters against a 44.8-character target — it has learned
where a molecule ends. By heavy-atom count (seed 0, whole 1,000-molecule test split,
`tools/g2s_report.py`):

| heavy atoms | n | graph valid | graph exact | SMILES valid | SMILES exact |
|---|--:|--:|--:|--:|--:|
| 0–10 | 39 | 0.8462 | **0.5385** | 0.8718 | 0.4359 |
| 11–15 | 117 | 0.7949 | **0.4957** | 0.7692 | 0.2821 |
| 16–20 | 231 | 0.8095 | **0.5238** | 0.7316 | 0.2814 |
| 21–30 | 400 | 0.7850 | **0.4875** | 0.6400 | 0.1975 |
| 31+ | 213 | 0.5681 | **0.2160** | 0.4554 | 0.0892 |

It degrades only past 31 atoms, where the node budget also bites. §5's rule holds — these columns are
not an arm comparison. **Attribution is confounded**: the stop-token fix and the instruct+chat switch
landed in the same build, so the base column is a floor, not a baseline. Separating them needs a
base-weights run with `answer_eos` on, which nothing requires.

**The ceiling question stays open and is cheap.** How much of 0.4193 is the 15 % share rather than the
capability: a g2s-only cell at the same horizon (`configs/probes/007_g2s_specialist.jsonc`, on instruct
weights) differenced against the graph cell. Not run.

### 7.2 Property prediction — ROC-AUC

From `tools/notation_probe.py --checkpoint`, which drops the union of truncated rows across arms so
every cell scores the identical 500 molecules.

| task | base graph | base SMILES | gap | instr graph | instr SMILES | gap |
|---|---|---|--:|---|---|--:|
| BACE | 0.8263 ±0.0513 | 0.8670 ±0.0092 | −0.0407 | 0.8303 ±0.0390 | 0.8599 ±0.0208 | −0.0296 |
| BBBP | 0.7137 ±0.0125 | 0.6883 ±0.0172 | +0.0254 | 0.7061 ±0.0237 | 0.6973 ±0.0333 | +0.0088 |
| HIV | 0.7587 ±0.0317 | 0.7869 ±0.0663 | −0.0282 | 0.7618 ±0.0373 | 0.7575 ±0.0449 | +0.0042 |
| SIDER | 0.8495 ±0.0030 | 0.8424 ±0.0061 | +0.0070 | 0.8586 ±0.0044 | 0.8449 ±0.0037 | +0.0138 |
| Tox21 | 0.7999 ±0.0162 | 0.8109 ±0.0015 | −0.0110 | 0.8373 ±0.0136 | 0.8231 ±0.0078 | +0.0142 |
| **five-set mean** | **0.7896** | **0.7991** | **−0.0095** | **0.7988** | **0.7966** | **+0.0023** |

| leg | SMILES − graph | sd | t(2) |
|---|--:|--:|--:|
| base 2× | +0.0095 | 0.0201 | 0.82 |
| instruct | **−0.0022** | 0.0126 | −0.31 |

Neither is significant (t critical at df=2 is 4.303). HIV moves most across the swap and is also the
set whose seed variance tripled at the doubled horizon (§8.4), so it is the least settled row.

### 7.3 Structural probes — exact match

The first nine are in-mixture; the last two are held out.

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

Every gap keeps its sign across the swap except `stereo_potential`, the weakest of them. The held-out
pair is at floor-level absolute accuracy — the ordering is the finding, not the score. ClinTox
zero-shot, measured on base weights only: graph 0.4046 ±0.0147, flat 0.5550 ±0.0769 — below chance on
the graph arm, and the reason the adaptation measurement matters more than the zero-shot one there.

### 7.4 Generation

> **These ChEBI-20 numbers are an in-mixture diagnostic and must not be placed beside a published
> row.** Three things separate them from a ChEBI-20 result, and they compound to **0.095 BLEU-2** on
> the instruct graph arm: they are a 500-row sample rather than the split (+0.012); they use
> `evaluate/captions.py`, whose tokenizer is a regex and whose METEOR is the exact-match stage alone,
> rather than the published protocol (+0.014); and they are scored on the 2,889 molecules this build
> admits rather than the benchmark's 3,300, the missing 411 being the large and multi-fragment end
> (+0.068). Measured the publishable way — whole split, published metrics, excluded molecules charged
> as misses — the same six checkpoints give graph **0.3290 ±0.0028** and SMILES **0.3533 ±0.0044**
> BLEU-2. See `molecules/TODO.md` §6 and `molecules/chebi_score.py` / `molecules/chebi_lit_metrics.py`.
>
> The table below stands as what it is: one instrument, applied identically to every arm and horizon
> here, and therefore a valid comparison *between our own columns*. The gap column is unaffected — the
> coverage charge is 0.068 on the graph arm and 0.072 on the flat one.

Only the two `instr` columns are a comparison; the `base` columns were measured with no stop token.

| task | base graph | base SMILES | instr graph | instr SMILES | gap |
|---|---|---|---|---|--:|
| `chebi20/bleu2` | 0.1937 ±0.0100 | 0.1975 ±0.0083 | 0.4257 ±0.0021 | 0.4567 ±0.0095 | −0.0309 |
| `chebi20/bleu4` | 0.1354 ±0.0121 | 0.1383 ±0.0100 | 0.3078 ±0.0025 | 0.3396 ±0.0096 | −0.0318 |
| `chebi20/rouge_l` | 0.3064 ±0.0116 | 0.3118 ±0.0078 | 0.5050 ±0.0018 | 0.5334 ±0.0077 | −0.0285 |
| `chebi20/meteor` | 0.4704 ±0.0262 | 0.4768 ±0.0222 | 0.4941 ±0.0016 | 0.5253 ±0.0065 | −0.0312 |

Captioning roughly doubled on both arms, and the flat arm's lead grew to 4–7 seed-sds. It is the one
family where the flat arm holds a consistent edge at the resolution available, and the expected
direction: a caption is text about a molecule, and the flat arm's molecule is already text.

### 7.5 Property 1, Property 2, leakage

**Property 1 holds 15/15 at the floor.** Every (set, seed) cell has `margin_spread_max` at or below its
own `margin_control_max` — the spread over ten relabelings of a molecule is not larger than the spread
the same run measures over ten re-runs of the identical input, 2 to 4 quanta of the margin. The chat
wrapper did not perturb it. (The verdict compares two maxima of the same grid-valued noise and allows
one quantum above the control; a strict `<=` there is a coin flip, and read 11/15 on the base-weights
trunk for exactly that reason.)

**Property 2:** `base_exact` reports every backbone weight equal to base Llama's exactly on all six
cells; the graph machinery is additive and removable. **Leakage** passes on all six: the stereo tag
ablation moves `stereo_assigned` by 0.04–0.06 against a line of 0.96, inside the band. No cell scored a
molecule its arm's training saw in another role.

### 7.6 What the adapter cost the backbone

`evaluate/text_behaviour.py` is the validator, `tools/text_behaviour_all.py` submits it,
`tools/text_report.py` reads it. 48 fixed prompts (`evaluate/text_probes.json`), greedy, 256 new
tokens, ~1.5 GPU-h for all six cells.

**The control is exact.** The backbone is frozen, and on a **single-node** graph every structural bias
is identically zero — `SPDBias` multiplies by `(spd > 0)` and `MagneticBias` masks `i == j` — so a text
prompt with adapters off *is* base Llama, with no second checkpoint resident. Both halves are pinned by
tests. It is also the logit comparison `base_exact` cannot make: there is such a batch, and it is a
single-node graph.

| | graph on | graph off | SMILES on | SMILES off |
|---|---|---|---|---|
| **answered with a molecule caption** | **0.139** ±0.012 | 0.000 | **0.021** ±0.021 | 0.000 |
| characters | 643.8 ±64.4 | 1014.8 | 706.7 ±37.1 | 1014.8 |
| new tokens | 141.0 ±16.6 | 217.1 | 176.0 ±9.9 | 217.1 |
| stopped inside 256 | 0.632 ±0.048 | 0.250 | 0.431 ±0.073 | 0.250 |
| one token or fewer | 0.000 | 0.000 | 0.000 | 0.000 |
| empty | 0.000 | 0.000 | 0.000 | 0.000 |

| divergence from the backbone | graph | SMILES |
|---|---|---|
| NLL of the backbone's own continuation | 0.5104 ±0.0082 | 0.5367 ±0.0201 |
| mean per-token KL(off ‖ on) | 0.2044 ±0.0030 | 0.2391 ±0.0284 |

**The register survived and the topic did not.** Two thirds of the mixture being single-token answers
did *not* teach the model to reply ` Yes` to prose. What it does instead is answer one general question
in seven with a ChEBI-20 caption — asked what the greenhouse effect is, a graph cell writes "The
molecule that is the simplest member of the class of benzenes … It has a role as a non-polar solvent".
Every other number in the table scores that answer as healthy. It is 6–8× the flat arm's rate, tight
across seeds, and 0.000 on the backbone.

**Averages were the wrong instrument.** The graph arm is *closer* to the backbone on both divergence
measures while failing worse and more specifically: smaller mean displacement, more severe tail. A
summary over 48 prompts cannot separate them; the per-prompt dump can, which is why the validator writes
one.

**Mechanism, offered as a hypothesis.** A text-only prompt is a single-node graph, and the graph arm
has never seen one: every training example it has is many-node. ChEBI-20 is the 19.3 % of its mixture
that maps a graph to prose, so a degenerate graph plausibly reaches for the nearest thing it knows. The
flat arm already trains on single-node items. If that is right the effect is about *graph-free prompts*
and not about task diversity — a molecule-instruction corpus contains no graph-free prompts either and
would not touch it. **Text replay, which does contain them, is the test, and it is §9.1.**

**Two readings that are not damage.** The adapter terminates on 63 % of prompts against the backbone's
25 % and writes 644 characters against 1015: the stop token taught on 45-character SMILES generalised to
ending *answers*. And the `system` condition — a turn no training example has ever contained — costs
almost nothing. The cross-arm floor is ~1 % on `chars_mean` (bf16 accumulation tipping a greedy argmax
deep into a generation, on two arms with different attention paths); the caption gap clears it by an
order of magnitude.

**What it indicates.** `generalist/PLAN.md` §6 specifies KL-to-base self-distillation on text-only
batches as the primary forgetting control, at a 15–25 % replay ratio. This measurement supports the
mechanism and narrows the target: the failure is specific to prompts with no molecule in them, which is
exactly where the KL teacher is free and exact. §6's "tune the ratio down until the suite moves" now has
a suite, and 0.19–0.24 nats/token is what the mechanism has to work against. The graph arm needs it more
than the flat arm.

---

## 8. What the earlier campaigns established

Kept for the findings, not the numbers. Each of these was measured on base `Llama-3.2-1B` in `Q:/A:`
formatting and superseded as a *result* by §7.

### 8.1 Property 1, on an arm that lacks it

Ten relabelings of every test molecule, the margin's spread across them, against the control the same
run measures (arm 2, base weights, 1×):

| set | graph spread | graph control | flat spread | graph AUROC spread | flat AUROC spread |
|---|---:|---:|---:|---:|---:|
| BACE | 0.2500 | 0.2500 | 6.2083 | 0.0029 | 0.1256 |
| BBBP | 0.2917 | 0.2500 | 11.5833 | 0.0028 | 0.0767 |
| HIV | 0.3333 | 0.2500 | 3.2500 | 0.0109 | 0.1399 |
| SIDER | 0.2917 | 0.2917 | 8.9531 | 0.0030 | 0.0305 |
| Tox21 | 0.2917 | 0.3333 | 5.1667 | 0.0080 | 0.0701 |

The flat arm is 10× to 40× wider on the raw margin, and the anneal *grows* its spread with its margins
(BACE 5.17 → 6.21) while the graph arm's stays on the quantum. The invariance survives the margins
growing underneath it, on an arm whose comparison does not.

### 8.2 Generalist against specialist

Arm 2 minus arm 1 (`molecules/PLAN.md` §8.4.8), paired within (dataset, seed), pooled over the nine
(set, seed) pairs of BACE, BBBP and HIV: against the specialist's last checkpoint (primary, since it is
free of a selection instrument shown to be near-blind) graph **+0.0100** winning 7 of 9, flat
**+0.0154** winning 6 of 9; against best-val graph −0.0099, flat +0.0024. **Training on every molecule
task at once neither helps nor hurts scaffold-split property prediction at a resolution three seeds can
see.** The free structural labels did not buy a measurable improvement and did not cost one.

Within arm 2 the split ran along one line — the flat arm ahead on property prediction (0.7969 vs
0.7795) and the graph arm ahead on the structural probes (0.9408 vs 0.9359) and both held-out topology
tasks — with the graph arm's probe seed spread twenty times tighter (±0.0003 vs ±0.0069). The
hypothesis was a SMILES pretraining prior; §8.3 is the test.

### 8.3 The notation ladder

**Zero-shot (0.1)**: base Llama, nothing trained, through the same margin readout: mean AUROC SMILES
0.4108, InChI 0.4713, SELFIES 0.5178, graph (adapters off) 0.5237. Nothing is consistently above
chance, so there is no ranking to select on, and the exposure ordering rests on the corpus-frequency
argument. SMILES is the outlier in the direction nobody proposed: lowest mean *and* most signal
(deviation 0.107), because its errors are systematic — 0.26 on BACE, 0.33 on HIV, strongly
anti-predictive. The backbone has a real disposition about SMILES strings and it points the wrong way.
A selection rule phrased on raw score would have chosen SMILES precisely because the backbone knows it
best.

**Trained (0.2)**: two more flat arms at arm 2's recipe and mixture, three seeds each; all twelve
annealed checkpoints re-scored on one instrument (`tools/notation_probe.py --checkpoint`, which excludes
the truncated rows the default validators do not).

| set | SMILES | InChI | SELFIES | graph |
|---|---:|---:|---:|---:|
| BACE | 0.8667 ±0.010 | 0.7888 ±0.021 | 0.8229 ±0.014 | 0.8185 ±0.019 |
| BBBP | 0.7086 ±0.014 | 0.6909 ±0.008 | 0.6940 ±0.011 | 0.7072 ±0.012 |
| HIV | 0.7291 ±0.010 | 0.7372 ±0.039 | 0.6896 ±0.019 | 0.7374 ±0.006 |
| Tox21 | 0.8100 ±0.016 | 0.7724 ±0.019 | 0.7677 ±0.007 | 0.7971 ±0.007 |
| SIDER | 0.8468 ±0.011 | 0.8351 ±0.001 | 0.8358 ±0.008 | 0.8332 ±0.008 |
| **five-set mean** | **0.7922** | **0.7649** | **0.7620** | **0.7787** |

| flat arm − graph | |
|---|---:|
| SMILES | **+0.0135** |
| InChI | **−0.0138** |
| SELFIES | **−0.0167** |

**The flat arm's advantage does not merely shrink as notation exposure falls — it reverses.** The edge
is a property of *SMILES*, borrowed from pretraining rather than earned by the representation. Bounds:
three seeds (the 0.030 swing clears the pooled ~0.021 resolution; the individual gaps barely do);
truncation biases in the result's own favour (the longer notations lose more training rows at
`max_length` 512, SELFIES up to 2.87 % on SIDER); ~11 % of the mixture is identical SMILES on every arm
(the four atom-level families, which only SMILES can index), diluting the contrast. Two build decisions
are part of what "SELFIES" means here: the notation rides on the *arm* so the arm-2 build hash stays
valid, and the SELFIES constraint set is `hypervalent` with the catch-all raised to 12, the one that
encodes and round-trips all 53,921 Tier-B molecules (`SELFIES_CONSTRAINTS`).

### 8.4 The doubled horizon

`budget_scale: 2.0` (§2) against the 1× campaign, both arms, three seeds: both arms improved (graph
+0.0109, SMILES +0.0069 on the five-set mean) and the flat lead narrowed from +0.0135 to +0.0095, but
the paired sd **tripled**, 0.0071 → 0.0201, with HIV on the flat arm the whole of the new variance
(per-seed [0.737, 0.732, 0.719] → [0.859, 0.774, 0.728]). HIV is exactly the corpus the fix moved most.
What was lost is resolution, not the gap: **anything built on this comparison needs more than three
seeds.** The held-out topology margin did not widen with the horizon.

### 8.5 Two defects, and the rule they share

**Stop token (found 2026-09-10).** No generative answer in any build before it carried an end-of-text
token — `TextGraphDataset.tokenize` has always taken `add_eos` and the adapter never passed it — so
`exact_match` on g2s and ChEBI-20 scored whether the model stopped. The graph arm was writing the
exactly-correct canonical SMILES as a **prefix** of its output 46.5 % of the time and running on to the
generation cap; it was reported as its worst result for two months. The defect was not arm-neutral: a
randomized SMILES input is a length cue the graph arm has nothing to match (33.5 % runaway against
98.7 %). Fixed by `answer_eos`, which moves `build_version`; the base-weights configs pin it `false` so
their artifacts stay reachable, and a test asserts the fix cannot land silently inside a reported
build. On base weights the right token is `<|end_of_text|>` (128001) and not `<|eot_id|>`, whose
embedding row sits at the reserved block's norm — initialisation, not training, and the output head is
frozen; on instruct weights all three answers move together to 128009, which is why the build reads
`tokenizer.eos_token_id` (`tools/stop_token_check.py` asserts all three on any backbone).

**Truncation (found 2026-09-06).** A flat arm is one node, so `max_length` 512 truncates a long SIDER
prompt from the right — taking the trailing answer with it — and `render` then supervises a
mid-molecule token. 162 of SIDER's 3,861 test rows on SMILES, 0 on the graph arm (`max_length` is per
node, and an atom is a handful of tokens). Evaluation now excludes truncated rows from every arm;
training swallowed 0.165 % of the SMILES draw, concentrated in SIDER — a disclosure, not a reason to
rebuild (`tools/truncation_census.py`). Found because `pos_rate` differed between arms scoring
identical molecules, which it cannot; it is now asserted rather than printed
(`check_arms_agree_on_labels`).

**The rule** (`molecules/PLAN.md` §9, now paid for five times): a quantity that is only ever *read* has
no error-detecting surface. `exact_match 0.0000` is the same number whether the model emits nothing, a
wrong molecule, or the right one and runs on. `tools/g2s_report.py` exists so the next zero comes with
its predictions attached.

### 8.6 Costs to plan with

Arm 2's ~82 GPU-h for six cells is obsolete in the direction that matters: most of it was evaluation
before the §D9 fixes. Measured after them, a flat-class cell at 1× is **~2.8 GPU-h** including both
milestones and its anneal, and six ran concurrently on twelve B200s in 1h27 wall. A graph cell at the
2× horizon on instruct weights is 8.4–10.2 h on two cards plus a ~1 h anneal. Four ranks made the arm-2
graph anneal *slower* than one would have (every new `(B, L, N)` triple pays a Triton autotune), so
price graph forks at one card until batches are bucketed. A walltime extrapolated across two factors at
once has come in 1.9× low before; confirm against the first cell's measured s/step.

### 8.7 Not run, and why

* **Graph+SMILES arm** (~60 GPU-h). Whether the structural channel carries anything the string does not
  already deliver. It forfeits Property 1, so it could never be the headline model; it would be a
  diagnostic. Deferred behind §9, which asks a more useful question of the same checkpoints.
* **`bias: none` control at the tuned recipe** (~6 GPU-h specialist form). Until it exists, "the graph
  arm" cannot be decomposed into structural channel versus extra adapter capacity. Still owed and
  cheap.
* **The 3B rung** (~170 GPU-h). `molecules/PLAN.md` §10.1's motivation is a recollection recorded
  nowhere, and §0 predicts no sign flip on molecules, since scaling a backbone does not create node
  text. It runs only after §9 has produced a mechanism worth carrying up a ladder, on
  `Llama-3.2-3B-Instruct` so the ladder varies size and nothing else, with `lr` re-screened at the new
  scale — the campaign's most expensive recurring mistake was inheriting it.
* **The g2s-only ceiling** (§7.1). Two cells on instruct weights, cheap, whenever the g2s number needs
  a bound.

---

## 9. The trunk as a base

**The question.** Every result above scores the trunk *on its own mixture*. What a trunk is *for* is
being started from. Three uses, in increasing distance from what it trained on, and each is a
measurement before it is a demonstration:

1. **A specialist on an in-mixture task.** Fork the trunk onto one property set and let it converge.
   The claim is that it reaches the from-zero specialist's score in a fraction of the specialist's
   steps.
2. **A specialist on a held-out task.** The same, on `bond_path`, `longest_chain` and ClinTox, which
   no training source touched. This is the adaptation number `generalist/PLAN.md` §3.3 has owed since
   the harness was designed.
3. **An assistant.** One fork onto the trunk mixture plus a curated set of free-form,
   graph-conditioned questions with natural-language answers. This is a case study: correctness is
   measured, conversational quality is read, and neither is optimised.

Every fork branches from the **annealed** graph checkpoint of `008`, not the stable one — that is the
model that is released, so it is the one whose value as a base matters, and it removes the
mid-schedule ambiguity a constant-LR checkpoint carries. The flat twin does not fork: the control for
each measurement here is a different *starting point*, not a different arm.

**Ordering.** §9.1 first, because it decides which checkpoints are the trunk of record for §9.2–9.4.
Then §9.2 and §9.3, which share a fork mode and an `lr` screen. §9.4 last, because its curated set is
the only new data and the only new code, and because its read depends on §9.1's answer.

### 9.1 Text replay in the trunk — screen, then decide

§7.6 found the trunk answering one graph-free prompt in seven with a molecule caption, and the
hypothesis is that it has never seen a single-node graph. The test is a trunk that has.

**The replay task.** `text/replay`, a new adapter emitting single-node graphs — a text prompt in the
user turn, exactly `build_flat_example`'s shape, so by Property 2 the forward pass is the base LLM's
and no model change is needed. **Targets are the backbone's own continuations**, generated once at
build time with adapters off, through the generation path `text_behaviour` uses: sampled at a modest
temperature rather than greedy, since greedy at 1B loops; 512 new tokens; and **only terminated
continuations are kept**, because the backbone stops inside 256 tokens on a quarter of prompts and an
unterminated target teaches truncation. That is KL-to-base self-distillation in its SFT form: the
teacher is exact and free, the corpus is a *prompt list* rather than a licensed answer set, and the
deferred `forgetting.py` with the token-level KL loss is the same idea with a better loss. Answer kind
`text`, `caption_rate` is its metric.

**Why the backbone's own answers and not a chemistry QA corpus.** Replay is forgetting control; its
target is *where the backbone already is*, and only the backbone can say that exactly. A chemistry QA
set is on-topic by construction, so it reinforces "prose in, molecule talk out" — the direction of the
defect — and a caption-rate drop under it could not be told apart from learning that corpus's style.
It would also bring facts about named molecules the model cannot ground in a graph. Chemistry
conversation belongs in §9.4, where every fact is verified.

**The prompt source.** Human-written, first-turn, English prompts from openly licensed instruction
sets — OpenAssistant's first user turns and Dolly's instruction field are the two — 5–10k in all,
stratified roughly in thirds: general knowledge and science, writing and reasoning tasks, everyday
instructions. On top of that, a deliberate slice of **chemistry in prose with no graph** — questions
about named compounds, reactions, lab practice — because that is the nearest neighbour of the failure
and the case the trunk most needs right: asked what caffeine is, it should write prose and not a
ChEBI caption. Deduplicated by exact match and 8-gram overlap against the 48 `text_probes.json`
prompts, which stay a clean test and are never trained on in any form. Same chat format as every
other example: user turn, assistant turn, no system turn.

**The screen.** One graph cell, seed 0, `008`'s recipe and horizon, with `text/replay` at **15 %** and
the four molecule blocks scaled by 0.85 (every block keeps its ratio to the others; nothing is thinned,
since the horizon is unchanged and the molecule corpora simply draw less often). The re-warm rule does
not apply — this is a fresh trunk, not a mixture change mid-run. Annealed exactly as `008`. ~25 GPU-h.

**Verdict, pre-registered.** Against `008` seed 0's annealed checkpoint:

* `caption_rate` on `text_behaviour` **≤ 0.03** (from 0.139), and `kl_mean` lower. That is the
  effect the mechanism predicts; anything short of it says graph-free prompts were not the whole
  story.
* The molecule suite within noise: five-set property mean within one seed-sd of `008`'s spread; every
  in-mixture probe within ±0.02; g2s `exact_match` within 0.03. Replay at 15 % costs 15 % of the
  molecule gradient, and the price of that is the number to report beside the caption rate.

**Both pass** → the three-seed graph campaign with replay (`009`, ~65 GPU-h) becomes the trunk of record
and §9.2–9.4 fork from it. **Either fails** → `008`'s three graph cells stay the trunk of record, the
result is reported as the bound on what SFT-form replay buys, and the token-level KL loss moves up the
deferred list. Either way the screen is one cell and one reported table, and the ratio is the WSD knob
it was always meant to be — tunable later without a rebuild.

**Result — the text is bought outright, and where it is paid for is the whole finding.**

The screen ran as written, and neither branch of the rule is what happened. Replay in the *mixture*
fixes the text and costs molecules at every share measured. Replay in the *decay* fixes the text just
as completely and costs nothing three seeds can distinguish from zero. Every number below is against
**the same seed's own annealed checkpoint**, which is the only matched control: seed 0's spread is not
seed 1's.

| | caption, plain | caption, system | KL(off ‖ on) | chars on | property mean Δ | g2s Δ |
|---|---|---|---|---|---|---|
| trunk (`008` graph, 3 seeds) | 0.139 ±0.012 | 0.118 | 0.204 | 644 | — | — |
| replay 15 % in the mixture (`009`, s0) | **0.000** | **0.000** | 0.103 | 967 | −0.0170 (−8.2 sd) | **−0.074** |
| replay 8 % in the mixture (`011`, s0) | **0.000** | **0.000** | 0.115 | 974 | +0.0052 | −0.036 |
| replay 15 % in the anneal (3 seeds) | **0.000** | **0.000** | 0.134 | 942 | **−0.0024 ±0.0076** | −0.008 |
| replay 40 % in the anneal (3 seeds) | **0.000** | **0.000** | 0.114 | 920 | −0.0016 ±0.0056 | −0.029 |

**The text bar is cleared several times over, at every share.** `caption_rate` is 0.000 on both prompt
forms in all five runs, from 0.139, and `kl_mean` falls by a third or more. The adapter-on answer
length returns to the adapters-off length (1015 characters), which says the terminate-early habit §7.6
read as harmless was also a symptom: it was the molecule register leaking into prose, and replay
removes both at once. The mechanism §7.6 proposed is confirmed — the failure was about graph-free
prompts, and one pass of them fixes it.

**The molecule cost is entirely a property of *where* the replay goes.** At 15 % of the mixture the
five-set property mean falls 0.0170 (−8.2 seed-sd), g2s `exact_match` 0.432 → 0.358, and the damage is
concentrated where the corpora are largest and the passes fewest: HIV −0.067, Tox21 −0.041, SIDER
−0.025, while BACE *rises* 0.033. That is 15 % of the molecule gradient removed for the whole run. The
same 15 % taken out of the 1,114-step decay alone costs −0.0024 ±0.0076 on the property mean and −0.008
on g2s — a null at three seeds, and seed 0 (−0.0109) was the outlier of the three, not the signal.

**Halving the mixture share does not halve the cost — it moves it.** At 8 % the property mean comes
back to +0.0052 and every probe clears ±0.02, with BBBP and BACE *up* (+0.059, +0.042) against HIV and
Tox21 still down (−0.032, −0.026). Read against a single run's own spread rather than the trunk's
seed-sd, that is a null and not a gain: the property sets tolerate an 8 % share. What does not come
back is g2s, still −0.036.

**g2s is the invariant casualty, and it orders the four runs by total replay seen.** −0.074 at 15 %
of the mixture, −0.036 at 8 % of it, −0.029 at 40 % of the decay, −0.008 at 15 % of the decay. Nothing
else in the suite is monotone in replay; graph-to-SMILES is the longest generation the trunk does and
the one that pays first when the molecule gradient is thinned.

**40 % is not better than 15 %.** It costs the same on the property mean and three times as much on
g2s (−0.029, at the pre-registered 0.03 bar) for no further text gain — 0.000 cannot go lower. The
decay is short enough that a sixth of it is already a full pass of graph-free prompts.

**Verdict.** `008`'s three graph cells stay the **trunk of record**, and the **reportable anneal
becomes the replay anneal at 15 %** (`replay_anneal15_graph_s{0,1,2}`). The 8 % mixture run is the
only other candidate that clears the probes, and it loses on the one axis that separates them — g2s
−0.036 against −0.008 — while costing a three-seed campaign to adopt. This is the cheap answer as
well as the correct one: one extra anneal per seed, about a GPU-hour, against the 65 GPU-h campaign a
new trunk would have cost, and the trunk's own checkpoints and hashes do not move. §9.2 and §9.3 fork
from the replay anneal; §9.4's assistant slice is a second entry in the same anneal's `add` list.

**One honest caveat about the bar.** "Within one seed-sd" was written against the trunk's three-seed
spread, 0.0021 on the property mean. A single anneal's own run-to-run spread is 0.0076 — larger than
the bar it was being judged against. The bar as pre-registered is tighter than the measurement it
applies to, so the paired three-seed mean is what the verdict rests on, and by the letter of the
pre-registration the anneal-only runs "fail" a bar that no rerun of the control would pass either.
That is a defect in the bar, stated here rather than quietly widened.

### 9.2 In-mixture specialisation — steps to the specialist's score

**Design.** `adapt` forks (`fork.py`, D6's third mode) with `allow_in_mixture_task: true` — the flag
exists for exactly this control, and its warning is correct: on an in-mixture task the number is partly
how recently the task was sampled, which is why the trunk's anneal, and not the stable checkpoint, is
the parent. One task per fork, both starts (`parent` = the annealed trunk, `base` = the instruct
backbone), identical config, three seeds. The mode asserts the two legs differ only in their starting
weights, so the from-base leg *is* the matched from-zero specialist at this recipe — same LoRA surface,
same bias set, same batch, same schedule. No number from `molecules/PLAN.md` §8.4.8 is differenced
against directly; those specialists ran base weights, a cosine schedule and their own batch.

**Tasks.** BACE, BBBP, HIV — the headline three. Tox21 and SIDER are not anchor-comparable and stay
out.

**Recipe.** Fresh optimizer, 10-step warmup then constant `lr` (`Schedule.training`), `budget_steps`
sized so the from-base leg converges — the specialist curves put that near 1,000 steps at a specialist
batch — `eval_steps 50`, which is the resolution of every steps-to-target number reported.
`tokens_per_step` is set to the specialist's batch and not the trunk's, so a step here is a specialist
step; the fork records its own value. **`lr` is screened first**, on BACE seed 0 only, both starts,
over {3e-5, 1e-4}: a fork from warm weights plausibly wants the lower rate, and the fingerprint rule
means both starts take the same one. The screen picks by the from-base leg's end-of-budget score — the
fair choice for the baseline, and conservative for the claim.

**What is reported, per task.** Both curves on one axis; the from-base leg's end-of-budget score as the
target; the step at which each leg first reaches **95 %** of it, and their ratio. Two pre-registered
absolute thresholds as well, so the read does not depend on where the base leg happens to end: the
specialist's three-seed last-checkpoint graph values from §8.4.8 (BACE 0.8133, BBBP 0.6882, HIV 0.7336).
The trunk's zero-step score is the first point on its curve and is the "as it is" number.

**Cost.** 3 tasks × 2 starts × 3 seeds = 18 legs at ~1 GPU-h each on one card, plus the screen: ~22
GPU-h. Confirm against the first leg.

### 9.3 Held-out adaptation — the measurement the plan has owed

**Design.** The same `adapt` forks without the flag, on the three held-out tasks (§4). The refusal on
an in-mixture task is the point of the mode and stays.

**Recipe.** As §9.2, with the same screened `lr`. `budget_steps` ~2,000 for the two topology tasks
(a specialist reaches 0.99 on `longest_chain`, and the from-base leg has to get there for the ratio to
mean anything) and ~1,000 for ClinTox. Every leg scores the `held_out` split only; the target names
`held_out`, never `test` (D7.4).

**Pre-registered thresholds.** `bond_path` and `longest_chain`: exact match **0.50** and **0.90**.
ClinTox `FDA_APPROVED`: ROC-AUC **0.70**. Plus, as in §9.2, 95 % of the from-base leg's end score and
the full curves. The trunk's zero-shot scores (§7.3: 0.076, 0.078, and ClinTox below chance on base
weights) are the first points on its curves.

**What would be a result either way.** The trunk crossing 0.50 on `longest_chain` in a small fraction
of the base leg's steps is the transfer claim the held-out pair's 3.3× zero-shot ordering has been
promising. The trunk *not* being faster on ClinTox is just as reportable: it separates structural
transfer (the traversal family, where the bias channel is the answer) from label transfer (a new
endpoint, where nothing in the graph channel is the answer), and that is a cleaner sentence about what
the architecture carries than a uniform win would be.

**Cost.** 18 legs, the topology ones twice as long: ~30 GPU-h.

### 9.4 The assistant — a curated set, one fork, a case study

**What it is.** Around ten thousand graph-conditioned exchanges in free-form natural language, across
as many phrasings, registers and shapes as can be had — a chemist's shorthand, a student's full
sentence, an instruction, a scenario, one fact or three — with no question template recognisable across
the set. What varies is not only *which fact* is asked for but *what the person wants done with it*:
reported, compared, checked against a claim they brought, used to decide something, written into a
record, explained, summarised for an audience. Some of those asks rest on a false premise the reply has
to correct, some need a fact the molecule's sheet does not carry and the reply has to say so, and a few
are underspecified and the reply is a question back.

Two shapes are in scope on purpose, because both strengthen the correctness read rather than weaken
it:

* **Format instructions.** "Answer as JSON", "one word", "a numbered list", "a sentence that starts
  with the ring count" — the format is part of the brief, the answer follows it, and the verifier
  checks the format as well as the facts.
* **Few-shot demonstrations, as graph components.** Withdrawn once and reinstated on the one condition
  the withdrawal named. The original axis asked the *writer* for a worked example, and a worked example
  is about a different molecule that nothing computed a fact sheet for, so no part of it could be
  verified — which is the one guarantee this set sells. What came back confirmed it: a demonstration
  asserting that the smallest ring containing atom 3 of 1,2-propanediol has 4 atoms, for a molecule
  with no rings at all.

  That objection is about where a demonstration comes from, not about few-shot. **Demonstrations are
  now drawn from the accepted set itself**, after the accept pass, so each one is a real molecule with
  an RDKit fact sheet that has already been through every filter here. The writer never sees them and
  never writes one. About **22 %** of examples carry 1–4, and the demonstration molecules enter the
  graph as their own **disconnected components** — which is also the only honest encoding, since a
  bond drawn between two molecules would be a chemical claim nothing computed. The prompt node carries
  a directed edge to each demonstration's node, so the pointer the question makes in words ("worked
  examples are attached — see those examples") is a real edge the structural bias can read. The
  distances that produces are the reason for building it this way:

  | from the prompt node | target molecule | a demonstration |
  |---|---|---|
  | its anchor node | — | 1 |
  | atom nodes | 1 | 2 |
  | Levi bond nodes | 2 | 3 |

  A demonstration reproduces the target's own pattern one hop further out, so *which molecule the
  question is about* is answerable from the prompt's SPD row rather than only from the text. Wiring the
  prompt straight to every atom in the graph would collapse both molecules to one distance and throw
  that away.

  Three rules keep a demonstration from making its example easier. It may not be the same molecule as
  the target — the partition key is the identity that matters, so two rows about one molecule cannot put
  the target's structure in its own context twice — and **it may not state one of the target's own
  facts**. That second one is the copy shortcut: a demonstration answering "3" in front of a target
  whose ring count is also 3 teaches that the answer is whatever the last example answered, which is the
  same defect as a question that states its own answer. Test-role targets take train-role demonstrations
  only, or two test rows in one context would stop being two independent measurements.

  The third rule is about yes/no answers, and it is the one that is easy to get wrong in the direction
  that looks careful. The pool skews yes, so drawing demonstrations on relevance alone hands a yes
  target a context of four yes demonstrations, and the answer becomes readable off the demonstrations
  without the molecule. Preferring the *opposite* polarity is the same shortcut inverted and just as
  readable. **Each demonstration's polarity is drawn 50/50, ignoring the target's** — a coin flip is the
  only arrangement that carries no information about the answer. Measured agreement over the composed
  set is 0.49.

  The prompt's edges into the *target* follow the example's own facts rather than the task name:
  `assistant.named_atoms_for` wires the prompt to the named atoms when every drawn fact is atom-scoped,
  and to the whole molecule otherwise. This is a corpus task whose rows mix the two, so the scope has to
  be decided per row. Atom labels are on for every row, atom-scoped or not, so the set carries one atom
  text convention rather than two.

The trunk mixture plus this set, annealed once. The result is a model a person can ask about a
molecule and get a sentence back.

**The one rule, in its strong form.** Every example starts from a molecule and a **fact sheet computed
by RDKit**, and **the model never states a fact.** A language model asked to count rings will be wrong
often enough to teach confident hallucination, which is the one thing this set cannot afford; handed
the sentence and asked to change its register, it is reliable. Earlier builds took the weak form of
this rule — write *from* the fact sheet — and spent three rounds of filters discovering that "from" is
not a constraint. The strong form is: the answer's claims are rendered by Python from the sheet, and
the model's whole job is to voice them. Facts are the thing the trunk was trained to know from the
graph and nothing else — this set re-skins trained capability, it does not add tasks:

* the nine Tier-A families' answers for the molecule (ring count and sizes, aromatic rings, functional
  groups present with counts, atom-level memberships, stereo potential and assignment);
* its canonical stereo-free SMILES (§5);
* any Tier-B labels it carries, with the endpoint named in words;
* its ChEBI-20 caption, if it has one.

**Not in the fact sheet, by construction:** anything from the traversal family (path lengths, chain
lengths, shortest or longest anything) and any trial-outcome or toxicity label — the held-out set (§4)
would otherwise be spent through the back door. A filter refuses any accepted question that mentions
them regardless of what the fact sheet held.

**Molecules** come from the **train role** of the §3 partition, sampled stratified by source and
heavy-atom count, one example per molecule, target **9,600 accepted examples**. The number is the fork
arithmetic below run backwards: it is what a single pass at a 15 % share needs to fill the standard
1,115-step anneal, so that the assistant's fork *is* the reportable anneal with one slice swapped in
and needs no control of its own. A shortfall after verification shortens the fork by the pass-cap
rule; it never raises the passes. A further **800 test-role** molecules go through the identical
pipeline to make the set's own test split: fresh molecules *and* fresh wordings, which is the right
read on whether the wrapping generalises.

#### The intent is declared, the truth is rendered, the model only voices

Three builds of this set failed the same way, and the failure is structural rather than a matter of
tuning. In all three the pipeline ran **facts → (question, answer)**: a model was handed a fact sheet
and asked to produce both halves, and a verifier then asked whether the two corresponded. That
correspondence is the thing no prompt can state — *invent a question answerable by exactly this
arbitrary subset of facts and by no others* — and every version of the set spent its yield there.
Templating the question moves the failure without removing it, because a template bank derived from
the fact schema has the schema's intent space: there are sixteen predicates on the sheet, so every
question either a harvest or a cross-product can reach reduces to *what is the value of predicate P on
atom A*. Six thousand phrasings of sixteen predicates is paraphrase variety, not instruction variety,
and a set built that way teaches a small set of lookups wearing costumes.

The pipeline that replaces it separates three things the earlier ones ran together in one model call.

**1. The intent is declared in data, before anything is written.** An intent is a row, not a prompt.
It names the **task** — report a value, compare two atoms or two molecules, check a claim the person
brings, decide under a stated constraint, fill a structured record, explain a consequence, summarise
for a named audience, triage a short list — the **facts** the answer rests on, an optional **twist**,
the **situation** it arises in, and the **style** the answer is asked for. Both halves of the example
descend from this one declaration, so they correspond by construction, and they do so from the opposite
direction to a template: the question is free prose and the *correspondence* is structural, rather than
the question being constrained and the correspondence hoped for.

The task axis is the one the earlier builds had no room for. It is orthogonal to which fact is
involved, and it is where instruction-following actually lives — what a person wants *done* with a
number, not which number it is. Three of its twists create behaviour the earlier sets could not teach
at all:

* **A false premise.** The person asserts something the sheet contradicts, and the reply has to correct
  it. Every example in the earlier builds carried a true premise, which teaches agreement.
* **An unanswerable ask.** The question needs a family this molecule's sheet does not carry, and the
  right reply says so. A set in which every question is answerable teaches a model to always answer.
  These draw only from **in-scope families absent from this molecule's sheet** — never from the §4
  held-out families, or the set would teach refusal on precisely what §9.3 measures.
* **An underspecified ask**, where the reply is a clarifying question and the person's answer to it
  resolves the example. This is the one place the set is multi-turn, and it is deliberate.

**2. The answer's content is rendered by Python, never written.** Given the molecule and the intent,
the renderer emits the answer's factual content as a canonical list of statements, and for a twist it
also renders the correction, the refusal, or the clarifying turn. Nothing about this step is
probabilistic. The fact sheet was already RDKit's; now the *reply's* claims are too.

**3. The model voices, and only voices.** Two calls, neither of which asks it to know any chemistry:

* it writes the person's turn from the situation and the ask — and it is **not shown the answer**, only
  what is being asked for, so a question cannot state its own answer;
* it re-voices the rendered statement list under the style brief — and it is shown the statements
  verbatim, so it is performing a string transformation on supplied content rather than retrieval and
  composition.

That second constraint is the whole point. Re-voicing supplied text in a register is what an instruct
model at this size does near-perfectly; composing a correct sentence about a molecule from a list of
predicates is where it errs. **Invented chemistry stops being a category that can occur**, because no
chemistry is being generated. What remains is a style failure — a dropped clause, a register missed —
which is benign where a confabulated benzene ring is not.

Few-shot is deliberately outside all three layers: it is drawn after acceptance, in the compose stage,
because a demonstration is another accepted row and there are none to draw from until the accept pass
has run.

**The writer is `gemma-4-31B-it`, loaded as an instruct model.** This matters more than any other
single choice here, and it was wrong for three builds — see below. Batches stay small and every example
carries its own style brief, because a model writing twenty in a row otherwise converges on one voice.

**Verification is mostly unnecessary now, and that is the measure of the redesign.** Nineteen
rejection reasons accumulated over three builds. Thirteen of them asked whether the question
corresponded to the facts, whether the answer overreached the sheet, or whether either invented
chemistry — and under a declared intent and a rendered answer none of those is expressible. They stay
wired as **assertions that must never fire**: one firing is a bug in the renderer or the situation
bank, not a row to discard, and it is reported as a build failure rather than counted as yield.

Three checks remain, and they are narrow because they are now checking a transformation rather than a
composition:

* **The person's turn names the ask's anchors** — the atoms and groups the intent declared — **and
  contains none of the rendered statements.** The writer was never shown the answer, so this is a
  guard against coincidence rather than against a behaviour, and it replaces `question_leaks_answer`,
  `question_drifts`, `question_mismatch`, `question_changes_subject` and `question_widens_scope`.
* **The re-voiced answer preserves the statement set.** Both sides of this comparison are text the
  pipeline produced, which makes it a far easier problem than reading a free composition: the failure
  mode is a dropped or merged clause, not a claim to adjudicate.
* **The format brief is met** — JSON parses against its declared skeleton, the list has the stated
  length, the one-word answer is one word. The skeleton is declared with the intent and rendered with
  the answer, so a format instruction can no longer be unsatisfiable for the facts drawn, which was the
  defect behind three of the four failures in the last hand sample.

**Where something still has to be judged, a model judges it.** Whether the reply is *responsive* to
the turn, and whether a re-voicing preserved meaning rather than merely vocabulary, are entailment
questions, and a 31B instruct model does them better than any pattern. The judge runs as a separate
pass over (statements, turn, reply), and it is a different task in kind from writing, so its blind
spots are not the writer's. **The judge is itself calibrated by hand on 100 rows** before its verdicts
are used, and its precision and recall are reported with the set — an unmeasured judge is an
unmeasured filter, which is the mistake this section has now made four times.

Then, unchanged: the held-out-family filter; question and answer length bounds; deduplication by 4-gram
Jaccard; a ceiling on how many examples share a style brief. The set ships as a JSONL carrying its
intent, its rendered statements, its brief and its writer on every row, so any example can be traced
back to the declaration it came from.

**The reported `verify` is statement preservation plus format**, and it remains the correctness metric,
so it remains an upper bound on correctness. It is a much tighter bound than before, because the
statements it checks against were computed rather than written.

#### What three builds established

**The writer was never the model it was labelled, and that alone accounts for most of the yield.**
Every row of v1–v4 carries `writer: "gemma-4-31B-it"`, but that string was only the `--writer` label.
The path behind it, `models/hf_models/gemma-4-31B`, is `google/gemma-4-31B` at sha `02e15e49` — the
**base, pre-trained** checkpoint. Worse, the `chat_template.jinja` beside it was added by hand and is
the **Gemma-3** template: `<start_of_turn>`/`<end_of_turn>` do not exist in the Gemma-4 tokenizer at
all, so the turn markers reached the model as ordinary text, the native `system` role was folded into
the user turn, and nothing emitted the `<|channel>thought\n<channel|>` prefix that suppresses
reasoning. The real markers are `<|turn>` (105) and `<turn|>` (106); the base `generation_config.json`
also stops only on `<eos>` where `-it` stops on `[1, 106, 50]`. Three builds of yield and rejection
numbers were produced by a base model prompted in a format it has never seen.
`tools/check_chat_template.py` is the guard, and it is now a hard gate in the writer launcher: it
renders one chat and refuses any marker that is not a single token id.

**What the correct writer is worth, measured.** A matched 2×2 over 400 shared examples — writer (the
base checkpoint vs the real `-it`, already on the cluster under `huggingface_cache/hub`) × task
(invent the question vs reword a seeded one), one repaired verifier throughout:

| writer | task | parse | verified | yield | distinct openings |
|---|---|---|---|---|---|
| base | invent | 0.980 | 169 | 43.1 % | 197 |
| base | reword | 0.943 | 251 | 66.6 % | 236 |
| `-it` | invent | 0.998 | 324 | **81.2 %** | 212 |
| `-it` | reword | 1.000 | 355 | **88.7 %** | 255 |

Fixing the checkpoint is worth +38.1 points on invention and +22.1 on rewording. Switching invention
for rewording is worth +23.5 points on the base model and only +7.5 on the real one. **Invention with
the correct writer beats rewording with the broken one**, so the finding that justified templating the
question — that the model cannot author one — was an artifact of the checkpoint and does not survive
re-measurement. `unsupported_claim`, the invented chemistry that worried this section most, goes 13→0
and 4→0 with the correct writer; `question_mismatch` goes 34→2. The templating step is therefore
dropped rather than repaired: it was solving a problem the writer no longer has, and it capped the
intent space at the sixteen predicates while doing so.

**Containment accepts about one wrong row in five.** Two hand reviews of v1 put the residual defect
rate at 8/37 and then 5/32 past every automatic check, always the same shape — the fact stated
correctly and the sentence *around* it wrong. The eight are worth keeping, because the rendered-answer
design exists to make each of them unreachable rather than caught:

- a law of chemistry invented to justify the number ("1, because there is 1 ether and all ethers
  contain exactly 2 oxygen atoms");
- an atom index read as an atomic property ("the atom is chlorine, which has the atomic number 17" —
  17 was its index, and the coincidence makes it worse, not better);
- a per-atom negative widened to the whole molecule ("atom 14 is not part of an ether" answered as
  "this molecule contains no ether group", of a molecule that has one);
- a yes/no opener contradicting its own sentence ("**Yes**, atom 8 … is **not** part of a hydroxyl
  group");
- a self-contradiction the sheet invited, by carrying "(0 means it is in no ring)" on a *nonzero*
  ring size: "has 6 atoms. Since it is in no ring, the ring size is 0";
- a JSON key naming the wrong family — `{"ring_count": 0}` written from a fact about the smallest
  ring containing one atom, which teaches that a molecule with three rings has none;
- sheet talk in a phrasing the filter did not have ("as explicitly stated in the facts");
- a question that stated all three of its facts, under a form not on the leaky list.

Every one of the eight is a sentence *composed* around a fact. None of them can be written by a model
that is handed the sentence and asked only to change its register, which is the argument for the
rendered answer in one line.

**Four rules, and they outlive the pipeline that taught them.**

1. **A filter's rejection log is evidence about the filter before it is evidence about the writer.**
   This section has now found filters discarding correct rows on five separate occasions. The last
   four were found by hand-reading rejections during the 2×2, and together they were most of what the
   rejection table was measuring: `question_drifts` was gated on the seed question being *present*
   rather than on reword mode, so it scored invented questions against a question their writer never
   saw; `_POLAR_RE` was `^`-anchored, which killed the yes-fact leak exemption for every brief that
   puts a preamble first — **66 of 69** leak rejections read by hand were false positives; a
   `_clauses` name collision silently swapped in the wrong splitter; and `_contains_yesno` read
   polarity over whole *sentences*, so one "not" negated every fact in a compound answer — **17 of 18**
   `facts_missing` rejections read by hand stated every fact they were given. Repairing those four took
   the two correct-writer arms from 76.9 % and 86.0 % to 81.2 % and 88.7 %.
2. **Fix the fact sheet before the filter where you can.** Two of the eight defects above were provoked
   by the sheet's own wording — a "(0 means it is in no ring)" gloss carried on a *nonzero* ring size,
   and "1 of its rings **are** aromatic" copied verbatim. A writer copies what it is handed, so sheet
   wording is a correctness surface rather than a style one. Under the rendered design this
   generalises: the renderer's wording is now the answer's wording, and it is the only place a phrasing
   defect can enter.
3. **A check worth keeping is one narrow enough to be certain.** An "atom census" subject first matched
   any "how many atoms", which is the commonest shape in the set (`ring_size`). Cut back to the two
   phrasings unanswerable under every reading — a formula or a mass, and a *plural* element count — it
   caught 40 rows with no false positive found on reading.
4. **A format axis that suppresses prose and a containment test that requires prose cannot both be free
   variables.** They were designed independently and contradicted each other, and the same collision
   produced the last hand sample's dominant defect: "answer as a numbered list" drawn beside exactly
   one fact, against a rule demanding two items, in 8 % of examples. The intent now declares the format
   skeleton *and* the statements together, so an unsatisfiable brief cannot be drawn.

**One cause was in the build rather than the writer, and it survives into the new design as a rule on
the intent.** `select_facts` sampled a multi-fact draw freely across the sheet, so one example could
carry an ether count and two unrelated aromatic-ring facts — and what came back was "There are 2
ethers. Atom 20 (O) is not in an aromatic ring, and atom 10 (C) is also not in an aromatic ring.
Therefore, there are 2 ethers." Each fact is true and contained, and the "therefore" is a derivation
nothing computed. Asked for several unrelated facts in one answer a writer will either weld them with
a false connective or list them as though they were related, and neither is a sentence worth learning.
**An intent's facts share a family or an atom with its pivot**, and the draw shrinks to one fact where
the sheet offers no such neighbour.

**The rejection table never had a category for "stated the chemistry wrong."** Grouped by what actually
went wrong, the 1,681 v2 rejections come out: question authoring 668, instruction following 661, answer
overreach 175, duplicates 177. Telling the writer all ten of the verifier's rules changed acceptance
from 83 to 83, because there is no wording of a rule that makes an unstatable constraint statable. That
reading was right about the constraint and wrong about the remedy: it led to templating the question,
which caps the intent space, where the right move is to stop asking a model to author *and* ground the
same sentence. The intent grounds it; the model authors it; neither does both.

**The template bank is retired and deleted.** `question_templates.json` and `assistant_templates.py` produced 5,943
templates over 59 signatures and lifted the v3 set from 34.9 % to 56.8 % accepted. Two findings retire
it. The gain was against a broken writer, and the 2×2 above shows the correct writer reaching 81.2 %
with no templates at all; and the bank's intent space is the sheet's sixteen predicates however many
phrasings wrap them, which is the ceiling the whole set kept hitting. What is worth keeping from it is
the discipline, not the artifact: **defects concentrate instead of scattering**, so the situation bank
and the renderer below are both small, readable, committed files — one bad line there is a few hundred
bad rows, and it is also one line that can be read.

Harvesting beat writing on the axis that mattered then — a hand-written bank would be thirteen stiff
frames in one voice — and one of its failures is a warning the new build inherits directly. Two gate
bugs were **silent**: a SMILES-fragment test keyed on "an uppercase letter and a bracket somewhere in
the token" also matched the slot name `{groups0}` and emptied every family with a group slot, and
deduplicating globally rather than per signature let one Tier-B corpus starve the other four. **A bank
that silently loses a family looks exactly like a bank that never had one.** So the situation bank and
the renderer report per-task and per-family coverage on every build, and a build that cannot ground an
intent fails loudly rather than falling back.

#### Building the ~10k set

**Volume.** The fork arithmetic below, run backwards: 0.15 of ~56 examples/step over the 1,114-step
decay is **~9,340 draws**. The set is therefore **9,600 accepted train examples at one pass** rather
than 2,400 at four — same draws, no repetition — plus **800 test-role** examples through the identical
pipeline, which is fresh molecules *and* fresh wordings.

**The order of work.** Each step gates the next, and the reason for the order is that a change to an
earlier step invalidates everything written after it.

1. **The renderer**, with the fact sheet it renders from. Deterministic, tested against RDKit, and the
   one place a phrasing defect can enter the answer.
2. **The situation and task banks** — the personas, the eight task types, the four twists. Small enough
   to read end to end, and read before anything is written.
3. **The intent sampler**: molecule × task × twist × facts × situation × style, with the pivot rule
   above, the format skeleton declared alongside the statements, and per-cell coverage reported.
4. **The two writer calls**, behind the chat-template gate.
5. **The judge**, calibrated on 100 hand-read rows before its verdicts count.
6. **Accept, then compose**, in that order — few-shot draws from accepted rows, so a defect admitted at
   step 6 propagates into every demonstration that uses it.
7. **A 100-row hand audit of the composed set**, stratified by twist, task and format, reported with
   an interval. This is the number that says whether the design worked; the `verify` rate is not it.
   The draw is even over the *twists* and the set is not — a twist is a small share, so a draw that
   mirrored the population would reach `needs_clarification` once and a defect that is universal
   within a twist would read as one unlucky row. Each drawn row therefore carries the share of the
   population its cell stands for, and both rates are reported: the rate over the draw answers "is any
   cell broken", the population-weighted rate answers "what fraction of the training data is bad".

**In the registry.** `mol/assistant`: kind corpus, `answer_kind text`, `verify` = statements preserved
and format met, `max_new_tokens 160`, `passes` **1**. Scored on its own test split by `verify` pass
rate, by the judge's verdict, and by the usual text metrics as a secondary read.

**The fork.** An `anneal` from the trunk of record, with `mol/assistant` added to the parent mixture at
share **0.15** — the one harness change this plan needs is an anneal that accepts an added task (the
`admit` planner already does the mixture arithmetic; what differs is that this fork *decays* rather
than re-warms, since the reportable models are all annealed and the comparison below should differ in
data only). Same active parameters as the anneal, biases included, so the only axis that moves is the
data. The pass cap and the share set the horizon, and the arithmetic is the point of choosing them
together: 9,600 examples × 1 pass = 9,600 draws, at 0.15 of ~57 examples/step, is **~1,120 decay
steps** — the standard anneal's 1,115, which `decay_steps` pins. The earlier plan reached the same
draw count as 2,400 × 4; a set this size reaches it without showing the model any example twice, which
is the better trade wherever the writing budget allows it. A higher share means a shorter fork or
repetition returning. `validate` confirms nothing is thinned, and confirms the fork's step count
matches the reportable anneal's.

**The control is the reportable anneal itself.** Same parent, same length, same schedule, same
active parameters, the parent mixture and nothing else — it already exists for every seed (§7). The
assistant is differenced against it on every molecule validator, so what is measured is the cost of
the 15 % slice and nothing else.

**What is measured, and what is only read.**

| | instrument | read |
|---|---|---|
| correctness | `verify` pass rate on the 800-example test split, broken down by task, twist, fact family and format | measured, reported, not optimised |
| defect rate | 100 composed rows, stratified by twist, task and format, read by hand and reported with an interval | measured — this is the number the design is judged on |
| what the wrapping cost | every §7 validator, against the reportable anneal | measured |
| general text | `text_behaviour`: `caption_rate`, length, stop rate, KL | measured — the assistant is the fork most exposed to §7.6's failure |
| how it reads | ~30 hand-written free-form prompts about test-role molecules, generations dumped by `tools/show_generations.py` into a report | read, quoted, not scored |

Three seeds, one fork each, ~1 GPU-h a fork on one card plus its validators; the writing pipeline is
two writer passes and a judge pass over ~10k examples.

**What it is not.** Not markdown, not an instruction-following benchmark, and not a claim about
capability the trunk did not already have. Multi-turn appears in exactly one place — the clarification
twist — and nowhere else; the set is otherwise single-turn. If it answers in a sentence that fits the
question, states the fact the graph determines, says so when the graph does not determine it, and does
not caption the greenhouse effect, it has done what a case study is for.

#### What went wrong, and the six shapes it took

Three smoke builds of 340 rows, then the full build, with every rejection read by hand after each of
the first three, a hundred *accepted* rows read after each of the last two, and the judge's 104
refusals read once the set was final. That found twenty defects. Not one of them was a writer defect, and not one was found by a rate: the rejection log
cannot show a row it let through, and `verify` passes at 1.000 on a set full of questions that do not
fit their answers. Reading rows is the instrument; everything below came out of it.

Ordered by class rather than by the build that turned it up, because the classes recur and the builds
do not.

**Class 1 — the reply owes something no statement enumerates.** Five instances, and the most
expensive class by a distance. The accept pass, the judge and `compose._reverify` all check that each
*statement* survived the re-voicing, and a statement is a sentence about the molecule. Everything else
a reply owes is invisible to all three at once.

| what the reply owed | how it went missing | rate before the fix |
|---|---|---|
| the **verdict** on a `decide` | stated every fact, never answered "does this one meet that?" | 3 of 156 rows |
| the **refusal** on an `unanswerable` | `answerable: False` reached the prompts and stopped there | 0 of ~800 — no backstop existed |
| the **correction** owed to a false premise | the verdict was given instead, in one word | 92 rows |
| the **gloss** on an `explain` | appended to the reply text, named nowhere, so the voice prompt never asked | 621 of 670 (92.7 %) |
| the **decision**, to the judge | `PRESERVED` is defined over the statements, so a reply that never decided preserved everything — and one that *did* decide was scored as an addition | penalised correct rows, passed broken ones |

The fix is the same each time: promote the thing to a first-class field on `Render`, name it in the
voice prompt, check it in the accept pass, excuse it to the judge. `explain` is the one worth naming,
because the gloss is the *entire* difference between `explain` and `report` and without it the task
had silently become a duplicate of another one. After the fix: 279 of 279 re-voiced rows carry it.

The rule, stated once: **ask of any check what the reply owes that the check does not enumerate.**

**Class 2 — a render that commits to content constrains the format draw.** The format is drawn after
the render, against `FORMATS`, so any shape or content the renderer has already fixed has to be
excluded there. Three instances:

- `_fill_record` writes its reply as JSON, and a prose brief over one answered "Does this molecule
  contain a nitrile?" with `{"fg_presence_nitrile": false}`. A render with a skeleton now takes only
  the JSON brief.
- A `decide` ask poses **two** questions — what the value is, and whether it meets the constraint —
  with two different answers. One word carries one of them, and the render gives the verdict, so the
  word answers the question that was not asked first. Where the constraint runs against the value it
  reads as its opposite: *"whether atom 6 (C) is part of a halogen"* → **"Yes"**, over a statement
  saying it is not. **239 rows, 144 of them contradicting.** `decide` is off `ONE_WORD_TASKS`.
- The first guard on that one excluded only `decide` over a *false premise*, on the reasoning that the
  premise is what makes it two questions. It caught 92 and left 147. The number of questions is the
  rule; the twist is a symptom. `check_claim` over a false premise is genuinely one question and one
  word answers it with the correction inside — "No", "Three", "Six".

**Class 3 — polarity taken from the sheet's wording rather than from the property.** `_clause(fact)`
renders a fact as the sheet states it, so it already carries the *value's* polarity, and `_negate`
flips that. Neither is "the positive form", and reading them as if they were broke three things:

- `_decide` built its constraint from `_clause`, so on every fact whose value was "no" the constraint
  came out negated while the verdict had been computed for the positive: *"Yes, that one qualifies.
  Atom 9 (C) is not in an aromatic ring."* About **60 % of `decide` rows**. `_polarised(fact,
  positive)` is the fix.
- `_negate` looked for "is not", "are not" and "contains no", and the HIV family writes its polarity
  into the verb — "It shows no activity against HIV replication." The wrapper produced the double
  negative "it is not the case that it shows no activity", the person's turn came out as "I believe
  this molecule does not show no activity", and the one-word reply to that was "Correct", which means
  nothing. The verb forms are handled, and the family has an `ASK_PHRASES` entry so the ask commits to
  neither polarity.
- `_negate` on a count swaps the fact's own digit into the fact's own sentence — and a `ring_size` of
  0 is spelled *"atom 24 (O) is in no ring"*, with no digit anywhere in it. The substitution was a
  silent no-op, so the false premise came out as the statement **verbatim** and the reply told a
  person their correct belief was wrong. **39 rows, 5.7 % of the `false_premise` twist**, all of them
  `ring_size`. This is the only defect the pipeline ever produced that *asserts* rather than omits,
  which makes it the worst of the twenty. `_negate` now falls through to the wrapper when the
  substitution changes nothing, and `premise_is_not_false` refuses any claim identical to a statement.

**Class 4 — a field nothing reads is a feature that does not exist.** One instance, and it survived
every stage. `needs_clarification` renders a four-turn exchange into `turns`; `question_text` returned
`row["question"]`, the underspecified opening turn alone. So the graph builder asked

> *Is that atom part of a ring? Please answer in one word.* — **"No"**

with an answer about atom 2, which the question never names. Built, rendered, filtered, verified, and
dropped at the consumer. Nine of twenty-five rows named an atom the question did not; the other
sixteen are worse, because they answer silently and the only learnable behaviour is to guess.
`question_text` assembles the exchange now, and `assistant_compose` stores the assembled form on a
demonstration for the same reason — a shot drawn from a clarified row would otherwise put an
unanswerable question in front of the target, answered. The assembly happens at the consumer, so a
composed set still re-verifies as the accepted set it came from. **When a twist adds a field, grep for
who reads it.**

**Class 5 — a draw with one cell in it, or none, or one that cannot fail.** Per-cell counts, not
totals:

- `unanswerable` drew its missing family from what a sheet happens to lack, which was `caption` **31
  times in 32** — one sentence, then deduplicated away. It was the wrong refusal to teach as well,
  since describing a compound is something the trunk can do. `OFF_SHEET_FAMILIES` adds eleven
  properties no 2D graph determines: melting point, pKa, IUPAC name, a synthesis route, an LD50, where
  to buy it.
- The count threshold was `max(0, value + choice(-1, 0, 1))`, which on any count of 0 or 1 gives "at
  least 0" — a decision that cannot come out no. **25 of 156 `decide` rows**, every one a guaranteed
  yes, visible in the verdict balance as 100 : 56. The floor is 1 and the balance is 81 : 75.
- A `check_claim` or `decide` may not pivot on the canonical SMILES. "I believe its canonical SMILES
  is COc1ccccc1N1C(=O)…" writes the whole molecule into the question and the reply copies it back, so
  the row is answerable without the graph at all. **88 rows.** `render.UNPIVOTABLE` refuses the draw
  and `sample_intent` takes a different task.
- `_record_key` keyed on family plus atom index, and the families with no atom index are the ones a
  molecule has several of — an `fg_count` per group, a `tier_b/tox21` per assay. Three assay facts all
  wanted the key `tier_b_tox21`; the dict kept one and the other two had nowhere in the JSON to be
  stated. The key now carries the assay or group name, with an index as a last resort.

**Class 6 — the writer's paraphrase of the ask changes the question.** One instance, found last and
the only one the rendered-answer design does not make unreachable. The inversion took content out of
the writer's hands, but the writer still writes the *person's turn*, and a turn is where a question
can quietly become a different one. Two `ring_size` facts enter as two phrases and come back as
"the size of the smallest ring containing atom 14 (C), atom 15 (N), and atom 10 (O)" — one clause,
three atoms, a ring nothing computed. The reply stays faithful to the statements, so every check that
compares reply to statements passes, and the row is unanswerable anyway. **7 rows of the set** against
821 that ask it correctly. `turn_asks_for_a_joint_ring` refuses it from the next build on.

The general lesson is the one this class is named for: the ask phrases are a *specification* of the
question, and the writer is free to paraphrase them into something that no longer matches. Anything
the ask phrases distinguish only by repetition is at risk, because repetition is exactly what a
natural paraphrase removes.

**Two more belong to no class and are recorded for completeness.** `_compare` joined its clauses as
`"{left}, whereas {right}."` and the sheet capitalises a clause opening with "It" or "The", so a reply
read "Atom 1 (C) is in no ring, whereas The smallest ring containing atom 19 (C) has 6 atoms";
`_lower_first` covers both connectives now. And `decide` on a count stated "I need that number to be
at least 5", which says *which* number only when the draw holds one — "How many ethers and how many
hydroxyls does this contain? I need that number to be at least 5" names neither, while the verdict
behind it was computed on the first. `COUNT_SUBJECTS` names the counted thing and falls back to the
old wording where it cannot, since an unnamed number is ambiguous but never wrong.

#### The rejections measure the filter first, the renderer second, the writer last

Rule 1 of this pipeline, confirmed on every build. Of twelve rejections read by hand after the first
smoke build, **ten were correct rows the filter was throwing away**; of eight after the second, six
were; of the final build's 104 judge refusals, **eighty were**. The rule has now held five times, at
sample sizes from eight to a hundred, and it has never once pointed at the writer first. What is new
since the redesign is where the *true* positives point: the renderer is now the only
place a content defect can come from, so a log that used to be evidence about a writer is evidence
about a filter first and a renderer second. The writer is the last place to look.

Seven filter repairs came out of those reads. Positional formats were being read as prose, so
"1. No\n2. No" was found to state neither statement it was given — a third of one build's rejections,
nearly all correct rows. A clause beginning with the connective that introduced it ("whereas atom 21
(O)") counted "whereas" as content and never merged into the predicate that followed, and the subject
was read as unnegated. A comma inside a coordination — "the NR AR LBD, SR p53, or NR AR assays" — was
read as a clause boundary, scattering one statement's content words over three fragments. "Nope" and
"Yep" were not read as answers. A statement with no answer token, which is to say a caption, was
checked clause by clause when no clause of a re-voiced paragraph carries 70 % of the paragraph.

The one structural change among them: **`render.answer_token(fact)` makes the renderer declare what a
reply has to carry**, so the accept pass stopped re-deriving the answer from the sentence — which is
how an assay named `SR ATAD5` came to be read as the number 5, and an atom index as a value.

**And a check gets rewritten against data, not against reasoning.** `states_the_verdict` took four
attempts, each corrected by real replies: a fact answered in two words ("no ether", "* SR p53:
inactive"), a list item opening "No sulfonamide", the reason trailing the decision ("it meets the
constraint because atom 11 (C) is **not** part of an ether"), the correction owed to a false premise
("Nope, only 2 rings are aromatic … But yes, it qualifies"), and a decision split from its answer by a
colon ("* Constraint met: / No"). What holds is weaker and does not try to identify the verdict at
all: **collect every span that bears on the decision and ask whether one of them agrees.** A reply
that decided has such a span; one that never decided has none; one that decided the other way has
spans that all disagree. Drop rate 43 % → 0.6 %, and the three rows left are two-item lists where the
verdict genuinely is not separable from the answers. Every version looked right in the abstract and
was wrong on data within a hundred rows.

**A coverage threshold gets swept, not chosen.** `states_the_gloss` read at the 0.7 of content words a
statement is held to refuses a reply that kept the explanation in half the words ("Generally, an
atom's part of a functional group if it's in the group's substructure"). Swept over the replies
written before the gloss was named and the replies written after, the share admitted runs 0.63, 0.27,
0.085, 0.079, 0.073 as the fraction rises from 0.3 to 0.7, while the share of *told* writers admitted
holds at 1.000 until 0.7, where it drops to 0.962. Flat across 0.5–0.6, so `GLOSS_COVERAGE` sits at
**0.5** — the loose edge, because a gloss has no polarity to invert and the writer is re-voicing a
fixed sentence, so a loose check admits a vague paraphrase where a tight one discards a correct row.
`declines()` is drawn wide for the same reason and the opposite stakes: a false rejection costs one
row in eight hundred, a false acceptance costs a hallucination.

#### Operating the pipeline

Six stages — `build` (CPU) → `ask` → `voice` → `judge` (GPU) → `accept` → `compose` (CPU) — driven by
`intent_pipeline.sh --out <dir> [--from <stage>]`. On one B200: build ~2 min/340 rows, ask 0.38 s/row,
voice 0.32 s/row, judge 0.25 s/row; ~11.9k rows is ~3.2 GPU-hours over the three writer passes.

Six things that cost time and are not obvious from the code:

- **A blocking `sbatch` piped to `tail` loses its exit status**, so a `&&` chain of stages runs on
  after a stage has failed. Cancelling a voice job mid-run let the judge run against an empty
  directory and write 150 empty verdict files, which the accept pass would have read as *every row
  unjudged*. Chain on the launcher's own status, never on a pipeline's.
- **A second build needs `intent_build --id-prefix`.** Ids restart at `train-00000` and batch files at
  `train-batch-0000.json`, and the ask, voice and judge passes all join on the id — so an unprefixed
  top-up silently reads one row's reply against another's statements. The two builds then merge by
  copying into one directory, and `accept` runs over the union so the dedup pool is the whole set.
- **The test split is read first.** Deduplication is greedy and first-wins against a growing pool, so
  whichever split is read first keeps its rows. The pool holds 44,088 train-role molecules against
  5,503 test-role, and the test split is the measurement; in plain sorted order a top-up's `b-train-*`
  batches come before `test-*` and take the slots, which cost the test split 94 rows against its
  target while the train split sat 700 over its own. `intent_accept.read_order` fixes the order.
- **Do not edit `render.py` or `intents.py` while a build is in flight.** The batches store the render,
  so a mid-flight edit leaves the writer working from one truth and the accept pass checking against
  another. Where a new field is a pure function of what the batches already hold, migrate rather than
  rebuild — a one-off script recovered the gloss off the end of 1,166 rendered replies, after which
  voice and judge re-ran. Rebuilding would have redrawn the molecules for nothing. The script was
  deleted once it had run; it was deterministic and single-use, and keeping it would suggest the
  migration is part of the pipeline.
- **A defect decidable from the stored render costs no GPU time to remove.** All three defects the
  final read found are decidable that way, so they were rejected in the accept pass and the set
  rebuilt from the existing judged batches — `accept` and `compose` are CPU-only and take minutes.
  Re-voicing would not have helped in any case: the brief is wrong, not the reply.
- **Keep the judged batches for as long as the set is live.** That last property is the whole reason
  three defects cost nothing to remove, and it is not free: `build`, `ask`, `voice` and `judge` are
  what a re-accept reads, and without them a filter fix means paying the GPU passes again. The merged
  set holds them as symlinks into the builds it was merged from, so tidying a superseded build away
  takes the current set's provenance with it. `accepted/` and `composed/` survive that — every row
  carries its own statements, facts, brief, skeleton and twist — but the ability to re-cut the set
  does not.

**Held for the next build, not done.** The sheet writes "It contains 1 amide(s)", and the `(s)`
reaches the data by the route I had ruled out — a `false_premise` claim is the sheet's own sentence
handed to the writer to put in the person's mouth, so it comes back as "I believe 1 stereocenter(s)
have a defined configuration", which nobody types. It is 0.6 % of written turns, all `check_claim` or
`decide`, and the accept pass drops them, which is cheaper than re-rendering 11.9k rows for seventy.
The renderer should stop producing the spelling: pluralise in `_clause` and widen the `_GROUP_RES`
pattern that parses the group name out of `\d+ ([a-z ]+)\(s\)`.

**Four tools exist only to be read by a person, and they are the instrument this section runs on.**
Nothing below is part of a build; each one takes a finished stage and prints it in the shape that
makes a defect visible, because every defect in this section was found by reading and none by a rate.

| | reads | prints |
|---|---|---|
| `intent_read` | batches, optionally `--asks`/`--voiced` | one render per (task, twist), with the writer's turn and reply beside the brief it was given |
| `intent_reject_read` | `accepted/rejected.jsonl` | rejections stratified by reason, `--reason X --n 12` — the rule is that a reason large enough to matter gets read before it is believed |
| `intent_audit` | `composed/` | a sheet stratified by twist, task and format, then `--mode score` over the labels written against it |
| `intent_calibrate` | the stages plus `judged/` | the judge's sheet and, after labelling, its precision and recall per axis |

One thing is deliberately left as it is. `needs_clarification` lands at about 1 % of the set, because
only `report` takes it and only a third of `report` draws are ambiguous enough to earn it. §9.4 wants
multi-turn rare, so that is the intended rate rather than a shortfall.

#### The set that was built, and what the audit measured

Two builds merged — 11,900 train-role draws and 3,000 test-role — carried through all six stages and
accepted over the union, so the dedup pool is the whole set.

| | |
|---|---|
| accepted | **10,640 of 14,900 (0.714)** — 9,873 train, 767 test, against targets of 9,600 and 800 |
| judge | preserved 0.987 / added 0.006 on the first build, 0.990 / 0.006 on the top-up; precision **0.231** on its 104 refusals, read by hand |
| demonstrations | 22.7 % of train rows and 19.8 % of test rows carry 1–4; 3,383 distinct rows used, max reuse 6 |
| demonstration polarity | agreement with the target **0.503** over 2,769 pairs — chance, which is the design |
| re-verified after composing | 10,640 of 10,640 |

Rejections, as a share of draws:

| reason | rows | share |
|---|---|---|
| `duplicate` | 3,073 | 0.206 |
| `one_word_over_two_questions` | 540 | 0.036 |
| `reads_like_a_data_sheet` | 188 | 0.013 |
| `statement_dropped` | 160 | 0.011 |
| `pivot_is_the_structure` | 90 | 0.006 |
| `judge_added` | 59 | 0.004 |
| `premise_is_not_false` | 42 | 0.003 |
| `clarification_not_ambiguous` | 41 | 0.003 |
| `judge_dropped` | 32 | 0.002 |
| `verdict_dropped` | 14 | 0.001 |
| `judge_unresponsive` | 13 | 0.001 |
| `refusal_dropped` | 5 | 0.000 |
| `turn_states_the_answer` | 3 | 0.000 |

`gloss_dropped` does not appear at all, which is what a fix that reaches the voice prompt looks like
from the accept pass.

**The audit: 100 composed rows over 48 (twist, task, format) cells, 25 rows per twist, read by hand.**

```
6/100 rows defective in the draw — 0.060 [0.028, 0.125] (Wilson 95 %)
      population-weighted       — 0.080

by twist   false_premise 4/25   none 2/25   unanswerable 0/25   needs_clarification 0/25
by class   question_mismatch 4   answer_wrong 1   question_leaks 1
```

Ninety-four rows are clean: every statement preserved, the question answered, nothing added. The six
that are not fall into exactly three kinds, all three now refused in the accept pass, and **all three
were then counted exactly over the set** — every one is decidable from the stored render, so the
audit's job was to *find* them and not to size them:

| | rows | share of the set |
|---|---|---|
| a false premise that is not false | 39 | 0.0035 (5.7 % of the twist) |
| `decide` under "one word" | 239 | 0.0217 (144 contradicting) |
| a SMILES claim copied back | 88 | 0.0080 |
| union | **366** | **0.0333** |

Two things in that are worth keeping.

**The stratification is what found them.** A draw that mirrored the population would have put ~88 of
its 100 rows in `none` and reached `false_premise` six times, and four defects in six rows is not a
signal anybody acts on. Twenty-five rows per twist made it 4 in 25. The earlier read, stratified by
(task, format) only, had sat on the same set and missed all three.

**The two rates disagree, and the disagreement is the finding.** The draw rate is 0.060 and the
population-weighted rate is 0.080, higher — because the two defects that landed in `none` belong to
the classes with the largest absolute counts, and `none` is 88 % of the set. A single rate over a
stratified draw would have understated the cost; a single rate over a proportional draw would have
missed the broken cell. Report both.

**What it cost to fix: no GPU time.** All three are decidable from the render, so they became
rejections and the set was rebuilt from the existing judged batches. Re-voicing would not have helped
in any case — the brief is wrong, not the reply, and the person's turn was written asking for one
word.

**The judge's precision is 0.23, and the miss is one cell.** The hundred hand labels written for the
judge went stale when the gloss fix re-voiced every reply, so it was calibrated instead on the thing
that survives a rebuild: all 104 rows it refused, read against the ask, the reply and its own stated
reason. Twenty-four were genuinely defective. **Eighty were correct rows thrown away** — 0.769 of
everything the judge removed, and the fifth time a filter in this section has quietly been discarding
correct rows.

| axis | refused | defective | precision |
|---|---|---|---|
| `judge_added` | 59 | 6 | 0.102 |
| `judge_dropped` | 32 | 11 | 0.344 |
| `judge_unresponsive` | 13 | 7 | 0.538 |

Thirty-nine of the 59 `judge_added` refusals are one row repeated: a `fill_record` over an
`unanswerable` twist, whose schema has a field for the property nobody can compute, answered
`{"ld50": null, ...}`. `null` **is** the decline — it is the only way to decline inside a schema the
ask itself fixed, and `intent_accept.declines` has counted it as one since the first build. The judge
reads it as a supplied value. That single misreading is 37.5 % of everything it threw away. Five more
are the prose version: on an `unanswerable` `report` the judge calls the refusal itself unlicensed
content, because it is never told the twist. Both are judge-prompt defects, and neither is reachable
from the accept pass.

Three more refusals are not verdicts at all but reasoning truncated at the token ceiling, cut off
mid-sentence having just talked itself round to the right answer. The judge's `max_new_tokens` is too
low to let it finish disagreeing with itself.

What it was right about is worth keeping: a lead that contradicts its own detail (6 rows — a
three-question `triage` enumerates its verdicts in the asked order, then names the atoms in a
different one), and a list that answers nothing (3 rows — asked whether side effects are reported in
two named classes, the reply bullets the two class names and asserts nothing).

Recall is not measurable on this build and will not be: the per-batch verdicts for the rows the judge
*passed* are not retained. Precision is the half that matters here anyway — the judge can only remove
rows, so its errors cost yield and cannot cost correctness. What the number says is that the
`--judged` deviation cost about 80 good rows out of 14,900, which is a price worth paying once and
not worth paying again with the same prompt.

**Reading the refusals found a defect nobody was auditing for: the question.** Eight of them are not
writer defects at all. Two `ring_size` facts enter the ask as two phrases — "the size of the smallest
ring containing atom 23 (O) and the size of the smallest ring containing atom 15 (C)" — and the
writer, paraphrasing that into something a person would type, compresses the repetition away:

    What is the size of the smallest ring containing atom 14 (C), atom 15 (N), and atom 10 (O)?

The shorter sentence is the natural one and it asks a different question, about a single ring holding
all three atoms. Nothing computed that ring; the statements are per-atom; no reply built from them can
answer it. Neither the statement checks nor the judge is positioned to see this — the reply is
faithful to the statements, and it is the *question* that moved — which is why it took reading the
refusals of an unrelated filter to find it. **821 rows ask it correctly with the clause repeated and 7
ask it this way**, 0.0007 of the set, the rest having been caught incidentally by filters aimed at
something else. Those 7 stay in the shipped set; `turn_asks_for_a_joint_ring` refuses them from the
next build on, keyed on the comma rather than on the atom count, because a turn naming several atoms
in one clause over a single `ring_size` fact is a person asking loosely and still answerable.

**The build's upstream stages are not retained.** What is kept is `accepted/` (the set and its
rejection log) and `composed/` (the shipped rows), and those are self-describing — every row carries
its statements, facts, brief, skeleton, twist and rendered reply, so the audit and the three exact
counts above can all be redone from the set alone. The per-batch `ask`, `voice` and `judged` stages
are gone, and with them the ability to re-run accept and compose cheaply over this build. That is the
one thing worth arranging differently next time: the render-decidable defects above were all fixed for
free *because* the judged batches were still on disk, and that is a property to keep deliberately
rather than by luck.

### 9.5 Checklist

- [x] Gates, build, cross-check, arm 2, the notation ladder, the doubled horizon, the stop-token and
      truncation defects, the instruct campaign, the general-text measurement — all landed by
      2026-09-12 and reported above.
- [ ] **§9.1** `text/replay` adapter; prompt source chosen and recorded; one screening cell; the
      verdict table; the trunk of record named.
- [ ] **§9.2** `lr` screen on BACE; 18 in-mixture `adapt` legs; curves and steps-to-95 % per task.
- [ ] **§9.3** 18 held-out `adapt` legs; curves against the pre-registered thresholds.
- [ ] **§9.4** renderer and fact sheet; situation and task banks; intent sampler with its coverage
      table; the two writer calls behind the chat-template gate; accept then compose; the 9,600 + 800
      set as JSONL; the 100-row hand audit; the judge's precision read over all 104 of its refusals;
      `mol/assistant` in the registry; the anneal-with-added-task fork; three forks; the four-row
      table and the read.
- [ ] Still owed from before, unchanged in priority: the `bias: none` control; the g2s-only ceiling;
      the `val` role shrink at the next rebuild.
