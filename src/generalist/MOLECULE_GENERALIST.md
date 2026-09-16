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

**What it is.** A few thousand graph-conditioned questions in free-form natural language, each with a
natural-language answer, across as many phrasings, registers and shapes as can be had — a chemist's
shorthand, a student's full sentence, an instruction, a scenario, one fact or three — with no question
template recognisable across the set. Two shapes are in scope on purpose, because both strengthen the
correctness read rather than weaken it:

* **Format instructions.** "Answer as JSON", "one word", "a numbered list", "a sentence that starts
  with the ring count" — the format is part of the brief, the answer follows it, and the verifier
  checks the format as well as the facts.
* **Few-shot demonstrations, in text.** One or two worked examples of the answer format inside the
  question, about a *different* molecule by name, and never carrying the query molecule's SMILES (a
  graph arm must not see one, §1). Several molecule graphs in one prompt is not what the prefix-node
  layout supports, so a demonstration is text and the example stays one graph.

The trunk mixture plus this set, annealed once. The result is a model a person can ask about a
molecule and get a sentence back.

**The one rule.** Every example starts from a molecule and a **fact sheet computed by RDKit**, and the
writing is done from the fact sheet. A language model asked to count rings will be wrong often enough
to teach confident hallucination, which is the one thing this set cannot afford; asked to *phrase* a
fact it already has, it is reliable. Facts are the thing the trunk was trained to know from the graph
and nothing else — this set re-skins trained capability, it does not add tasks:

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
heavy-atom count, one example per molecule, target **2,400 accepted examples**. The number is the fork
arithmetic below run backwards: it is what four passes at a 15 % share need to fill the standard
1,115-step anneal, so that the assistant's fork *is* the reportable anneal with one slice swapped in
and needs no control of its own. A shortfall after verification shortens the fork by the pass-cap
rule; it never raises the passes. A further **200 test-role** molecules go through the identical
pipeline to make the set's own test split: fresh molecules *and* fresh wordings, which is the right
read on whether the wrapping generalises.

**The writers are agents, and they only write.** The pipeline is Python end to end: it computes the
fact sheets, draws one style brief per example (register, form, length, how many facts, format
instruction or not, few-shot or not, whether to lead with the answer or the reasoning), writes batch
files of ~20 (fact sheet, brief) pairs, and runs the verifier on what comes back. An agent's whole job
is one batch file in, one JSONL of (question, answer) pairs out. Batches are small and every example
has its own brief, because a model writing twenty in a row otherwise converges on one voice. The
writer pool is several different models on rotating briefs — the variety comes from model × brief —
and the model and brief are logged on every row; no model writes more than a third of the set. About
125 batches for the target, a few hours of wall clock in parallel, and no GPU.

**Verification, before an example is accepted.** Every fact the answer uses is stored on the example
(`meta.facts`, a list of (family, canonical value)); the answer must contain each value in canonical
form (a number, a yes/no word, a group name, the SMILES), and any SMILES quoted must parse and
canonicalize to the molecule's. That containment check, plus the brief's format check where there is
one (JSON parses, the list has the stated length, the one-word answer is one word), is the task's
`verify` (D2), and it is also the correctness metric. Then: the held-out-family filter; question and
answer length bounds; deduplication by 4-gram Jaccard against everything already accepted, with a
ceiling on how many examples share a style brief; and a hand review of a 10 % sample — about 250 rows,
a morning — with every rejection reason logged. The set ships as a JSONL with its facts, its brief and
its writer on every row, so any example can be traced.

**In the registry.** `mol/assistant`: kind corpus, `answer_kind text`, `verify` = facts contained and format met,
`max_new_tokens 160`, `passes` **4**. Scored on its own test split by `verify` pass rate and by the
usual text metrics as a secondary read.

**The fork.** An `anneal` from the trunk of record, with `mol/assistant` added to the parent mixture at
share **0.15** — the one harness change this plan needs is an anneal that accepts an added task (the
`admit` planner already does the mixture arithmetic; what differs is that this fork *decays* rather
than re-warms, since the reportable models are all annealed and the comparison below should differ in
data only). Same active parameters as the anneal, biases included, so the only axis that moves is the
data. The pass cap and the share set the horizon, and the arithmetic is the point of choosing them
together: 2,400 examples × 4 passes = 9,600 draws, at 0.15 of ~57 examples/step, is **~1,120 decay
steps** — the standard anneal's 1,115, which `decay_steps` pins. A higher share means a shorter fork
or more repetition; the weight is as high as four passes allow and no higher. `validate` confirms
nothing is thinned, and confirms the fork's step count matches the reportable anneal's.

**The control is the reportable anneal itself.** Same parent, same length, same schedule, same
active parameters, the parent mixture and nothing else — it already exists for every seed (§7). The
assistant is differenced against it on every molecule validator, so what is measured is the cost of
the 15 % slice and nothing else.

**What is measured, and what is only read.**

| | instrument | read |
|---|---|---|
| correctness | `verify` pass rate on the 200-example test split, broken down by fact family and by brief (format, few-shot) | measured, reported, not optimised |
| what the wrapping cost | every §7 validator, against the reportable anneal | measured |
| general text | `text_behaviour`: `caption_rate`, length, stop rate, KL | measured — the assistant is the fork most exposed to §7.6's failure |
| how it reads | ~30 hand-written free-form prompts about test-role molecules, generations dumped by `tools/show_generations.py` into a report | read, quoted, not scored |

Three seeds, one fork each, ~1 GPU-h a fork on one card plus its validators; the writing pipeline is
CPU and agent time.

**What it is not.** Not multi-turn, not markdown, not an instruction-following benchmark, and not a
claim about capability the trunk did not already have. If it answers in a sentence that fits the
question, states the fact the graph determines, and does not caption the greenhouse effect, it has
done what a case study is for.

### 9.5 Checklist

- [x] Gates, build, cross-check, arm 2, the notation ladder, the doubled horizon, the stop-token and
      truncation defects, the instruct campaign, the general-text measurement — all landed by
      2026-09-12 and reported above.
- [ ] **§9.1** `text/replay` adapter; prompt source chosen and recorded; one screening cell; the
      verdict table; the trunk of record named.
- [ ] **§9.2** `lr` screen on BACE; 18 in-mixture `adapt` legs; curves and steps-to-95 % per task.
- [ ] **§9.3** 18 held-out `adapt` legs; curves against the pre-registered thresholds.
- [ ] **§9.4** fact-sheet builder; brief list and batch writer; agent runs; verifier and filters; the
      2,400 + 200 set as JSONL; `mol/assistant` in the registry; the anneal-with-added-task fork;
      three forks; the four-row table and the read.
- [ ] Still owed from before, unchanged in priority: the `bias: none` control; the g2s-only ceiling;
      the `val` role shrink at the next rebuild.
