# Molecules × GTLM — implementation plan

**Status:** M0–M2 complete (2026-08-30). **M3 is next, and it is scoped to BACE / BBBP / HIV**
(§1 Tier B, 2026-08-31). Written to run **before** the generalist trunk
(`src/generalist/PLAN.md` §7 names molecules as an admission-gate domain and leaves the
representation decision open — this document is that decision).

**Condensed 2026-08-31.** The Tier-A campaign generated ~90 runs across ten sweeps and this
document grew a blow-by-blow account of each. That narrative is preserved in git history
(`1e84c4a` and earlier) and is the place to look for how a conclusion was reached. What is kept
here is the conclusions, the decisions, the retractions, and the reasoning that still governs how
the campaign is run. Section numbers are unchanged from the long version because ~30 code and
test comments cite them.

**Provenance.** Repo-internal results are cited to their sweep. External anchors were read from
the papers named inline; §2 marks which are primary and which are secondary. Anything marked
*(estimate)* is arithmetic, not measurement.

## Configs

Every sweep is reproducible from `configs/`, one file per sweep, numbered contiguously:

| config | runs | what it settled | § |
|---|---:|---|---|
| `000_smoke` | 6 | plumbing, flex-vs-eager, s/it and peak GB | 3.1.2 |
| `001_m3a_bace_bbbp` | 18 | **M3a — Tier B, BACE + BBBP, both arms, 3 seeds.** Renamed from `001_bace` and rewritten 2026-08-31: the recipe was stale, the budget was ~756 steps, and the scope is now two datasets | 8 |
| `002_difficulty` | 18 | Tier-A difficulty screen at 1250 steps | 3.2.4 |
| `003_hardware_check` | 4 | H100/A100 run the flex compile path | — |
| `004_extended` | 20 | the screen at 2500 steps, matched pairs | 3.2.4 |
| `005_fgcount_ablation` | 3 | bias/question-node ablation — **result retracted** | 3.2.6 |
| `006_recipe` | 8 | **the recipe fix; `longest_chain` flips** | 3.2.7 |
| `007_tuned_ablation` | 4 | **the win is the bias, and the bias is SPD** | 3.2.8 |
| `008_m2_screen_graph` | 7 | M2 closing screen, graph arm at `bias_lr 1e-2` | 3.2.5 |
| `009_m2_screen_flat` | 5 | M2 closing screen, flat arm (its control twin) | 3.2.5 |
| `010_tier_b_smoke` | 2 | **first GPU contact for the Tier-B path** — plumbing only, no quotable number | 8 |
| `011_m2_loose_ends` | 7 | M2's two stated holes: a second seed on §3.2.8's `longest_chain` cells, and gate A2's never-run `stereo_tags: off` control — **running it is what exposed §3.2.10** | 8, 3.2.10 |
| `012_m3b_hiv_prep` | 2 | builds the two HIV artifacts (~2.6 GB graph) so 013 hits a warm cache — **written, not submitted** | 8 |
| `013_m3b_hiv` | 9 | M3b — HIV, both arms, 3 seeds, budget in *steps* not epochs. **Written, not submitted: gated on M3a's `tied_pair_fraction`** | 8 |
| `014_m2_rerun_molsplit` | 20 | ✅ **THE M2 CLOSING TABLE** — ten families × both arms on molecule-disjoint splits, one recipe, one dataset build. Supersedes `008`+`009`+the three families quoted from `004`. **5 graph wins / 2 ties / 2 flat / 1 void; every graph margin grew** | 3.2.5 |
| `015_m3a_budget_check` | 8 | does Tier B want a longer budget? 3× epochs on BACE/BBBP × both arms × 2 seeds. Exists because §8.2's inverted flag made "M3a is budget-limited" look true; this measures it instead of arguing it | 8.2 |
| `016_leakage_detector_restore` | 3 | **restores the suite's only leakage detector** — `stereo_assigned` off/on on the measured non-degenerate pool, with the pass line pre-registered at 0.774. **Written, not submitted: held pending a decision** | 9, 2.5 |
| `017_budget_check_oom_rerun` | 1 | ⚠️ **SUPERSEDED BY `019`, cancelled before producing a number** — carried `eval_steps 200` where `015` uses 50 | 8.2 |
| `018_budget_check_stall_rerun` | 1 | ⚠️ **SUPERSEDED BY `019`, cancelled before producing a number** — same `eval_steps` error | 8.2 |
| `019_budget_check_reruns` | 3 | three of the four graph cells `015` lost to infrastructure, re-run at 128G. **Results belong in `015`'s table**, not their own. Command line verified identical to `015`'s | 8.2 |
| `020_budget_check_rerun_bace_s0` | 1 | the fourth (`bace`/graph/seed 0), split off only because it was still running when `019` was submitted — it then OOMed at **97%** of training | 8.2 |

`014` **supersedes `008` + `009`**, which measured the same families on contaminated splits
(§3.2.10). It also folds the two files back into one: with `bias_lr` carried on the arm bundle,
there is no longer a reason to split the arms across configs. The `pool` widened from `bace,bbbp`
to all five corpora because the fix made the narrow pool *infeasible*, not because a wider pool was
wanted — `014`'s header carries the measured shortfall.

`008` and `009` are **one experiment in two files** — split only so each arm gets hardware sized
to it (graph peaks at 30.8 GB, flat at 11.7 GB). Neither is meaningful alone. `009` runs five
families rather than seven because 006 already holds the `fg_count` and `longest_chain` flat cells
at this recipe, and `bias_lr` provably cannot reach the flat arm (it instantiates no bias
parameters — `009`'s header carries the run-record proof).

Two things that are deliberately absent. The M2 canary's config was **deleted** (§3.2.3): it set
`node_position_mode`, which no longer exists, so it could only crash, and a config that cannot run
invites someone to try. The width analysis (§3.2.5) has **no config by design** — it re-scores
another sweep's checkpoints, so its per-run inputs are that sweep's outputs.

`results/` directories on disk carry **pre-renumbering** names (`001_smoke`, `004_difficulty`,
`006_extended`, `007_width`, `008_fgcount_ablation`, `009_recipe`). Map them by name, not number.

---

## 0. The trap this plan is built to avoid

Molecule property prediction is the one domain where GTLM's headline mechanism is **void**.
The thesis (`CLAUDE_CONTEXT.md` §1) is that graph-tokenizing pipelines destroy node text by
compressing it to one vector, and GTLM wins by not compressing it. An atom is `C`. There is no
text to preserve. On `ogbg-molhiv`, GTLM is a graph transformer with a 1B LLM bolted on,
competing against Uni-Mol and GEM — models pretrained on ~10M molecules **with 3D conformers**,
which GTLM has no channel for.

Framed as "beat MoleculeNet SOTA", this campaign fails, expensively, and produces a
citable-against-us result. Same shape as the rel-trial outcome (`project-reltrial-headline-result`).
`src/generalist/PLAN.md` §7 already writes the mitigation for relbench — *"score it on its own
terms"*. Molecules need the same discipline written **before** the GPU commit, not after.

**So what are molecules actually for?** Four things, none of which is a leaderboard:

1. **Free, exact, unlimited structural labels.** RDKit is a ground-truth oracle, which makes
   molecules a *generator* domain — single-pass, no overfitting, fresh eval sets, RLVR-eligible
   later. KGQA, TAG and relbench are all finite corpora.
2. **Structural diversity the mixture lacks.** Every other domain is sparse-directed (KGQA Levi),
   ego-net (TAG), heterogeneous-bipartite (relbench) or abstract (GraphQA). Molecules are small,
   dense, undirected, cyclic, chemically typed and homogeneous in size.
3. **An honest flat twin.** SMILES is heavily represented in Llama's pretraining, so the
   flat-serialization control is a genuinely strong opponent rather than a strawman.
4. **A property claim the leaderboard cannot touch** — permutation invariance over atoms (§6),
   which GTLM has by Property 1 and every SMILES LLM provably lacks.

**Target of the campaign:** beat our own SMILES flat twin; land in the InstructMol band at 1/7 the
parameters; **do not claim to compete with Uni-Mol or any 3D specialist.**

---

## 1. Benchmarks — three tiers

### Tier A — the structural diagnostic suite ✅ built and measured

RDKit-generated QA over real drug-like molecules (drawn from BACE + BBBP). This is the
`substructure` probe (`src/experiments/probes/README.md` Probe 2, *"Proxy for: molecules"*)
promoted from glued-together synthetic rings to real chemistry with a real opponent.

| Task family | Question | Ground truth |
|---|---|---|
| `ring_membership` | "Is atom 14 part of a ring?" | `atom.IsInRing()` |
| `ring_size` | "Smallest ring containing atom 14?" | SSSR |
| `ring_count` | "How many rings?" | SSSR |
| `aromatic_ring` | "Is atom 14 in an aromatic ring?" | RDKit aromaticity |
| `fg_presence` / `fg_count` | "How many carboxylic acid groups?" | SMARTS match |
| `fg_atom_membership` | "Is atom 14 in a nitro group?" | SMARTS match, atom-level twin |
| `bond_path` | "How many bonds separate atom 3 and 14?" | graph SPD — **held out (§4.1)** |
| `longest_chain` | "Longest unbranched carbon chain?" | graph traversal |
| `stereo_potential` | "How many atoms *could* be stereocenters?" | `FindMolChiralCenters(includeUnassigned=True)` |
| `stereo_assigned` | "How many stereocenters are *assigned*?" | CIP labels, from the SMILES `@`/`@@` tags |

Metric: exact match. 50/50 balanced where binary.

**The two stereo rows are a matched pair and the distinction is the point.** `stereo_potential` is
determined entirely by connectivity (deciding four subtrees are pairwise non-isomorphic) and the
graph arm should win it. `stereo_assigned` needs information that reaches the graph only as a
parity tag, and flat winning there is the suite confirming it measures what it claims. A suite in
which the graph arm wins everything cannot distinguish "the structural channel works" from
"something leaked".

**`stereo_tags` is a switch, not a global decision.** The **parity tag** (RDKit's
`CHI_TETRAHEDRAL_CW`/`CCW`) is standard input, not cheating — OGB's own `atom_to_feature_vector`
carries chirality as its second atom feature, so every GNN baseline receives it. `stereo_tags: on`
is the Tier-B and trunk default. The **CIP label** (`"chiral R"`) would be cheating: parity is
*information about the molecule*, the CIP label is *the answer to the task*. Never put it in node
text. Record `stereo_tags` in every run record.

### Tier B — MoleculeNet ⚠️ SCOPED 2026-08-31

Scaffold split (ours, `scaffold_split`), ROC-AUC. Built at M3 (`tier_b.py`); GPU runs pending.
Molecule counts are post-drop; *examples* are `(molecule, endpoint)` pairs after skipping absent
labels.

**Primary set — run these three, and nothing else, until they have reported:**

| Dataset | molecules | endpoints | examples | positives | scaffold split |
|---|---:|---:|---:|---:|---|
| **BACE** | 1513 | 1 | 1513 | 45.7% | 1210/151/152 |
| **BBBP** | 2039 | 1 | 2039 | 76.5% | 1631/204/204 |
| **HIV** | 41119 | 1 | 41119 | **3.5%** | 32895/4112/4112 |

**The reason for the scope is the anchors, not the cost** (2026-08-31). These three are the
only Tier-B sets with a *complete anchor ladder* — 3D specialist, graph-LLM at 7B, tuned-LLM at
7B, and a prompted-LLM floor (§2). That ladder is what converts one AUROC into a position in the
space of solutions. Tox21 and SIDER have a single anchor row between them, so a number on them is
uninterpretable in exactly the way §0 warns about. They are also all single-endpoint, so the first
Tier-B measurement does not simultaneously test the multi-endpoint routing machinery.

**Order them BACE → BBBP → HIV.** HIV is 27× BACE's size and the imbalance stress case: ~145
positives in its test split, so a handful of distinct margins decides the AUROC outright.
`n_distinct` and `tied_pair_fraction` are in the run record from the first run for exactly this.
BACE and BBBP settle whether the pipeline works; HIV is a stress test that follows.

**The scaffold split moves the base rate between train and test, and it moves it a long way**
(measured 2026-08-31, pinned in `tests/experiments/molecules/test_tier_b_examples.py`):

| | train | val | test | overall |
|---|---:|---:|---:|---:|
| BACE | 0.426 | 0.556 | **0.605** | 0.457 |
| BBBP | 0.822 | 0.549 | **0.524** | 0.765 |

BBBP's positive rate falls from 82% in train to 52% in test. Two consequences for M3. **AUROC is
rank-based and unaffected**, which is why it is the headline metric and the one every §2 anchor
reports. **`accuracy` and `f1` are badly affected** — a model that learns the training prior scores
near-chance accuracy on BBBP's test split while ranking perfectly well — so do not read those
fields as a health check on a run, and do not compare them across datasets. This is the scaffold
split doing exactly what it is for (a structurally novel test set), not a bug.

**Deferred, not cancelled:**

| Dataset | molecules | endpoints | examples | positives | why deferred |
|---|---:|---:|---:|---:|---|
| Tox21 | 7823 | 12 | 77864 | 7.5% | one anchor row; 16012 labels absent |
| SIDER | 1427 | 27 | 38529 | 56.8% | one anchor row; the widest and most skewed set |
| ClinTox | 1478 | 2 | 2956 | 50.6% | **held out permanently** (§4.1) |
| ESOL / FreeSolv / Lipo | 1128 / 642 / 4200 | 1 | — | regression | needs `numeric_text`; `validate()` refuses them |

**The cost of deferring Tox21 and SIDER is §4 arm 2, not the leaderboard.** They were what handed
the multi-task arm a large within-domain set for free (78k + 39k examples from under 8k molecules
each, because one example per `(molecule, endpoint)` pair makes multi-endpoint sets the bulk of
Tier B). Deferring them thins that arm considerably. Record it as a deferral so it is not quietly
dropped: **they re-enter at M3c if the primary three clear the gate in §8.**

`ogbg-molhiv` optionally, if the OGB leaderboard framing is wanted; same molecules as HIV under a
different split.

### Tier C — ChEBI-20 captioning (optional, unbuilt)

33,010 molecule–caption pairs; graph in, free text out. The only molecular task where the LLM half
does real work. Rank it third. Caveat to write down before running: ChEBI-20 captions are highly
templated and BLEU/ROUGE reward template matching, so a strong number there is weak evidence.

### Deliberately excluded from cycle 1

* **QM9 / quantum properties** — 3D-determined; without a 3D channel (§7) a guaranteed loss that
  says nothing about the architecture.
* **peptides-func / peptides-struct (LRGB)** — *"Where Did the Gap Go?"* showed most of LRGB's
  long-range gap dissolves under proper tuning. Keep peptides for the size-generalization protocol
  (§5), where they are a legitimate instrument.
* **Molecule generation from text** — the output is a SMILES string; the graph is on the wrong
  side of the model.

---

## 2. Anchors, and what "beating them" means

The rows below are not interchangeable and blending them produces a meaningless target. **Our
matched control is our own flat twin; everything here is an external anchor** measured by someone
else, at 7B, often with pretraining we do not have. Say so in any writeup, exactly as
`src/experiments/relbench/PLAN.md` §1 caveat 1 does.

### 2.1 How these baselines are trained — read before comparing

The field's default protocol is **pretrain once, then fine-tune a separate model per downstream
dataset**, with a task head and per-dataset hyperparameter search. Almost every row below is that.
This matters twice: it makes our **arm 1 (specialist, one model per task, §4)** the protocol-matched
comparison rather than a shortcut, and it means none of these numbers is a generalist result.

| Row | Protocol |
|---|---|
| Uni-Mol, GraphMVP-C, MolFM, MolCLR, GraphCL, KV-PLM, MoMu | SSL pretrain → **separate fine-tune per dataset**. Uni-Mol pretrains on ~209M conformations / ~19M molecules with 3D. |
| InstructMol-G / -GS | Two-stage. Stage 1: encoder + LLM frozen, projector trained on 264K PubChem caption pairs. Stage 2: **task-specific instruction tuning**, encoder frozen, projector + LLM via LoRA, 10 epochs. The paper states *"for each task, corresponding instruction templates are designed"*; it is **not explicit** about whether BACE/BBBP/HIV share one property-prediction model. It is certainly not one model across property prediction + reactions + captioning. |
| Llama-2-7B-chat, Vicuna-v1.3-7B (★ rows) | **LoRA fine-tuned**, not zero-shot. The row we target is a tuned baseline. |
| Vicuna-v1.5-13b-16k | 4-shot in-context, no tuning — hence the near-chance numbers |
| Galactica-6.7B / 30B / 120B | Prompted, no downstream fine-tuning |
| [Mol-LLM](https://arxiv.org/abs/2502.02810) | The genuine generalist: one model over a broad task suite (SELFIES + hybrid GINE/TokenGT encoder + Q-Former), trained with next-token prediction plus a structure-preference objective |

### 2.2 The primary three — MoleculeNet classification, scaffold split, ROC-AUC ↑

InstructMol (arXiv:2311.16208, **current version**, Table 2; 3 random seeds, scaffold splits).
Primary source, read 2026-08-31.

**Every number below is as compiled in InstructMol's Table 2**, not re-run by us and not read from
each method's own paper — so a row's own paper may quote a different figure under a different
split. The model links go to the method's own paper, for what it *is*; the numbers stay attributed
to the table that put them side by side.

| | BACE | BBBP | HIV | how it works | why it is in this table |
|---|---:|---:|---:|---|---|
| [DMP (TF+GNN)](https://arxiv.org/abs/2106.10234) | 89.4 | 77.8 | 81.4 | Dual-view pretraining: a Transformer over SMILES and a GNN over the graph, trained jointly with a consistency loss between the two views | Strongest row in the table, and the closest published thing to **our two arms fused**. Evidence that string-view and graph-view are complementary rather than redundant — which is the optimistic reading of §3.2.5's split |
| [**Uni-Mol**](https://openreview.net/forum?id=6K2RM6wVqKu) | **85.7** | **72.9** | **80.8** | SE(3)-equivariant transformer over **3D conformers** with a pair-distance channel; pretrained on ~209M conformations / ~19M molecules | **The ceiling, not the target.** It has an information channel we structurally lack (§7 is the drop-in that would close it). The anchor most readers will quote at us, so §0's framing exists for this row |
| [MolFM](https://arxiv.org/abs/2307.09484) | 83.9 | 72.9 | 78.8 | Multimodal foundation model over graph + text + knowledge-graph neighbours, contrastively aligned | Text *and* graph without an LLM decoder — the nearest non-LLM neighbour to GTLM's information set |
| [GraphMVP-C](https://arxiv.org/abs/2110.07728) | 81.2 | 72.4 | 77.0 | 2D GNN pretrained by contrastive + generative agreement with 3D views; **3D is used only in pretraining, not at inference** | Measures how much of the 3D advantage survives as a 2D-only model — i.e. the part of Uni-Mol's lead that is not inference-time geometry |
| [MolCA (1D+2D)](https://arxiv.org/abs/2310.12798) | 79.8 | 70.0 | — | Galactica LM + 2D GNN joined by a Q-Former cross-modal projector plus a uni-modal adapter | **The graph-tokenizer architecture our thesis argues against** (§0): the graph is compressed to a few query vectors before the LM sees it. The architectural foil for GTLM's prefix nodes |
| [KV-PLM](https://doi.org/10.1038/s41467-022-28494-3) | 78.5 | 70.5 | 71.8 | BERT over SMILES tokens interleaved with biomedical text, masked-token pretraining | The pure flat-string baseline at BERT scale — our flat twin's modality, with domain pretraining and without an LLM |
| [MoMu](https://arxiv.org/abs/2209.05481) | 76.7 | 70.5 | 75.9 | GNN + text encoder contrastively aligned on molecule–description pairs | Contrastive alignment as the alternative to prefix-token fusion: same two modalities, joined at the loss instead of in the sequence |
| [GraphCL](https://arxiv.org/abs/2010.13902) | 75.3 | 69.7 | 78.5 | Graph-only contrastive SSL with augmentations; **no text channel at all** | The structure-only floor — what topology alone buys on these endpoints. The published analogue of our `bias`-only ablation (§3.2.8) |
| [ChemBERTa-2](https://arxiv.org/abs/2209.01712) | 73.5 | 69.8 | 79.3 | RoBERTa over 77M SMILES, MLM + multi-task regression pretraining | A SMILES LM *specialised on chemistry* — roughly the ceiling our flat twin would approach if Llama had been pretrained on SMILES rather than merely exposed to it |
| [**InstructMol-G**](https://arxiv.org/abs/2311.16208) (7B + graph tokens) | **84.3 ±0.6** | **68.6 ±0.3** | **74.0 ±0.1** | 7B LLM + frozen pretrained 2D graph encoder; projector aligned on 264K caption pairs, then LoRA instruction tuning | **The band we should land in**, and the closest architecture to ours — graph tokens vs our prefix nodes, at 7× the parameters. Gate B2 is written against this row |
| [InstructMol-GS](https://arxiv.org/abs/2311.16208) (7B + graph + SMILES) | 82.1 ±0.1 | 72.4 ±0.3 | 68.9 ±0.3 | Same, plus the SMILES string in the prompt alongside the graph | **Our `+smiles` arm (§3.2), measured by someone else**: +3.8 BBBP but −5.1 HIV against -G. Task-dependent in both directions, which is exactly why it cannot be our headline |
| [**Llama-2-7B-chat**](https://arxiv.org/abs/2307.09288), LoRA | **74.8** | **65.6** | **62.3** | General-purpose 7B chat LLM, LoRA fine-tuned on SMILES-in-prompt property QA. **No graph channel** | **The row to beat, at 1/7 the parameters** — and the nearest published analogue of *our own flat twin*. If our flat arm lands far from this, the comparison is broken before the graph arm is read (gate B3) |
| [Vicuna-v1.3-7B](https://lmsys.org/blog/2023-03-30-vicuna/), LoRA | 68.3 | 60.1 | 58.1 | Llama-1-7B tuned on ShareGPT conversations, then the same LoRA protocol as the row above | Isolates **the base model at fixed protocol**: same size, same adaptation, 6 points below Llama-2. A caution that backbone choice moves these numbers as much as method does |
| [Galactica-6.7B](https://arxiv.org/abs/2211.09085) | 58.4 | 53.5 | 72.2 | Decoder LM pretrained on scientific text including SMILES; **prompted, no fine-tuning** | What domain pretraining alone gives with no adaptation — the gap to the LoRA rows is the value of fine-tuning at all |
| Galactica-30B | 72.7 | 59.6 | 75.9 | as above, larger | Together with 120B: **scale without adaptation is not monotone** (120B is 11 points *below* 30B on BACE). Do not read parameter count as capability when arguing 1B-vs-7B |
| Galactica-120B | 61.7 | 66.1 | 74.5 | as above, larger | |
| [Vicuna-v1.5-13b-16k](https://lmsys.org/blog/2023-03-30-vicuna/), 4-shot | 49.2 | 52.7 | 50.5 | In-context learning only, **no weight updates** | **The floor, and it is at chance.** Establishes that these endpoints are not solvable by prompting, so every other row is measuring adaptation rather than latent knowledge |

**Version drift, checked and resolved.** ar5iv still serves an earlier version of InstructMol with
InstructMol-G at 85.9 / 64.0 / 74.0 — materially different on BBBP. The numbers above are the
current arXiv version. Do not "fix" this table against the stale mirror.

### 2.3 Anchors for the deferred sets

Needed only if §1's deferred sets re-enter at M3c. ROC-AUC, scaffold split:

| | Tox21 | SIDER | ClinTox | how it works | why it is here |
|---|---:|---:|---:|---|---|
| [Uni-Mol](https://openreview.net/forum?id=6K2RM6wVqKu) | 79.6 | 65.9 | 91.9 | see §2.2 — 3D conformers, ~19M-molecule pretraining | The only anchor these three sets have. One row is not a ladder, which is why they are deferred |

**Provenance caveat — these three are secondary.** The Uni-Mol paper's own table could not be read
directly (OpenReview serves a bot-check page to the fetch path); they come from a table that
quotes it. Its BACE/BBBP/HIV column reproduces InstructMol's Uni-Mol row exactly (85.7 / 72.9 /
80.8), which is a consistency check but not a substitute. **Verify against the Uni-Mol PDF before
these appear in a writeup.** No LLM-baseline anchors exist for these sets at all — part of why
they are deferred. The regression sets have no anchors of any kind yet.

### 2.4 Two anchors not to calibrate against

* **[Mol-LLM](https://arxiv.org/abs/2502.02810)** (arXiv:2502.02810) reports BACE 80.5 and **BBBP 81.1**. The BBBP figure is far out of
  line with every other row and its protocol differs (multi-task generalist, SELFIES, hybrid
  encoder). Do not calibrate against it without reading its split.
* **The 2026 systematic survey** (arXiv:2604.16586) Table 4 quotes SIDER 0.847 and ClinTox 0.984
  from a KA-GAT/KA-GCN line. SIDER in the field usually sits near 0.62–0.68. Those are scraped from
  source papers, not re-run. The same survey's own finding is the useful one: **method rankings are
  unstable across evaluation protocols** — four different models win the six ADME endpoints under a
  time-based split.

**Captioning (Tier C, if built):** MolT5-Large ≈ 25.9 BLEU-2; 3D-MolT5 42.05 BLEU-2 / 34.16
BLEU-4. All use 3D or large-scale molecular pretraining we do not have.

### 2.5 Pre-registered gates

| Gate | Criterion | Status |
|---|---|---|
| **A1** | Tier A: graph arm ≥ +15 points over the SMILES flat twin on `ring_membership`, `ring_size`, `bond_path`. | **Still not met on the named families, but closer on clean splits** (§3.2.5, `014`): `ring_size` +12.9, `ring_membership` +4.1. `bond_path` is held out (§4.1) and cannot be measured. Note `longest_chain` clears the bar at **+16.0** — the threshold is achievable, it is the *named* families that are ceiling-bound: the graph arm is at 0.978–1.000 on both, so the gap is capped by flat's headroom, not by the graph channel. |
| **A2** | Tier A: graph arm **wins** `stereo_potential`; on `stereo_assigned`, chance under `stereo_tags: off`, high under `on`. | **First half PASSES on a properly molecule-disjoint measurement (2026-09-01, `014`):** `stereo_potential` graph 0.767 vs flat 0.724, **+0.043 (+2.2σ)** ✅ — up from +0.025 on the contaminated splits. **Second half is now UNMEASURABLE on this pool:** `stereo_assigned`'s test split collapsed to a single answer (§3.2.10.1), so both arms score 1.000 and the off/on contrast has no signal to carry. The 2026-08-31 `011` result (0.9928 with the parity tag → 0.8152 without) stands as the only measurement of it, on the old pool and old splits. **Re-running the off/on contrast requires a pool where the family is not degenerate** — `bace,bbbp,tox21,lipo`, now **measured** at 10 classes / base 0.732 on its test split, against 1 class / 1.000 on `014`'s (§9). `016_leakage_detector_restore` is written against it and pre-registers the pass line at 0.774; until it runs, A2's second half is *unverified on clean data*, not passed. |
| **B1** | Tier B: graph arm beats **our own flat twin** by ≥ 1 sd on ≥ 2 of the primary 3 classification sets. **This is the real result.** | **FAILS on 2 of 2 measured (§2.6, `001`).** BACE −0.017 (−1.0 sd), BBBP −0.033 (−2.5 sd) — the graph arm *loses* to flat on both, so B1 cannot be met even if HIV wins. Not a budget or scoring artifact (§8.2, §2.6). **§3.2.5 pre-registered this outcome.** HIV (M3b) still to run for the record. |
| **B2** | Tier B: within 3 AUROC of InstructMol-G on BACE/BBBP/HIV at 1B vs 7B. Aspirational, not a gate. | **Met on both measured sets, by both arms.** Graph 81.8 / 67.2 and flat 83.6 / 70.5 against 84.3 / 68.6; our flat *exceeds* the BBBP row. §2.6 |
| **B3** | Tier B: beat the Llama-2-7B-chat LoRA row (74.8 / 65.6 / 62.3) at 1B. If we do not, the flat twin will not either, and the domain is telling us the molecules are being read badly — check M1's round-trip test before blaming the architecture. | **PASSES on both measured sets** (§2.6). Flat 83.6 / 70.5 and graph 81.8 / 67.2, both above 74.8 / 65.6 at 1/7 the parameters. The molecules are being read fine; B1's failure is not a plumbing failure. |
| **P1** | §6: flat twin AUROC spread across randomized SMILES ≥ 2 points; GTLM spread < 0.1 (bounded by Property 1's 2.77e-5). | Pending M6 |
| **Null** | If the graph arm does not beat flat on **Tier A**, the *encoding* is wrong, not the architecture. Do not theorize about the model until the round-trip test and a `bias=none` control have both been run. | Both run. Round-trip clean (§3.2.1); `bias=none` measured (§3.2.8). |

### 2.6 M3a — the first Tier-B measurement (`001`, 18/18 COMPLETED, 2026-09-01)

BACE + BBBP, three arms, three seeds, `lr 3e-5 / bias_lr 5e-3 / lora_r 16`, 40 epochs (BACE ~1480
steps, BBBP ~2040). Test ROC-AUC, mean ± sd over seeds; the sd is **seed-to-seed training
variance**, since `scaffold_split` is deterministic and all three seeds share one split.

| set | our **flat** | our **graph** (`spd+magnetic`) | our **graph** (`bias: none`) |
|---|---:|---:|---:|
| BACE | **0.8357 ± 0.0207** | 0.8183 ± 0.0136 | 0.7833 ± 0.0013 |
| BBBP | **0.7048 ± 0.0168** | 0.6719 ± 0.0087 | 0.6929 ± 0.0118 |

| contrast | BACE | BBBP |
|---|---:|---:|
| graph(bias) − **flat** — *this is gate B1* | −0.017 (−1.0 sd) | −0.033 (−2.5 sd) |
| graph(bias) − graph(none) — *what the bias channel buys* | **+0.035 (+3.6 sd)** | −0.021 (−2.0 sd) |
| graph(none) − flat — *what adapter capacity alone buys* | −0.052 (−3.6 sd) | −0.012 (−0.8 sd) |

**Gate B1 is not met, on either set.** The graph arm loses to its own flat twin. This is a clean
negative and it is not a budget artifact: recomputed convergence (§8.2) puts every BACE peak at
0.39–0.75 of the budget and every BBBP peak at 0.03–0.54, with 2 of 18 runs genuinely still
improving. Nor is it a scoring artifact — `n_distinct` is 67–87 against 152/204 test examples and
`tied_pair_fraction` is 0.004–0.017, so the bf16 tie-collapse trap `evaluate.py` watches for did not
fire.

**§3.2.5 predicted this in writing, before M3 ran, and the prediction was the risky one.** It
recorded: *"BACE binding, BBBP permeability and HIV activity are pharmacophore/motif-driven, so our
own Tier A result puts the graph arm on the losing side of its own split at Tier B."* That is a
pre-registered prediction of a result against our own interest, and it held on both sets. It is the
strongest evidence in this document that §3.2.5's topology-vs-motif reading is a real mechanism
rather than a post-hoc story — and it is worth more than the gate would have been.

**The bias channel is load-bearing exactly where the mechanism says it should be.** On BACE the
structural bias is worth **+0.035 (+3.6 sd)** over the same architecture with the bias off — the
whole graph arm's showing on BACE is the bias, since `bias: none` sits 0.052 *below* flat. On BBBP
it is worth −0.021. So the channel is not inert (§3.2.6's retracted null stays retracted) and it is
not universally good: it pays on the set with more topological content and costs on the one with
less.

> **Two apparent anomalies here, both checked and both benign.** `bace / graph / none` has a seed sd
> of 0.0013 — one part in six hundred. The seeds are genuinely independent runs: their eval
> trajectories, `best_val_score` (0.676 / 0.683 / 0.689) and `bias_norm` all differ, so this is not
> a seed that failed to wire through; three runs simply landed within 0.0025 of each other on test
> while spreading 0.013 on val. Separately, `bace / flat / seed 2` and `bace / graph+bias / seed 2`
> report **byte-identical** test ROC-AUC (0.814764). That is quantisation, not a duplicated record:
> BACE's test split is 92 positives × 60 negatives = 5520 pairs, so ROC-AUC can only take ~5520
> values and collisions are ordinary. The two records differ everywhere else (`bias_norm` 0 vs
> 41.15). Neither needs action; both are recorded so the next reader does not re-derive them.

**Read against §2.2's anchors, the picture is much better than gate B1 alone suggests** — because
the arm that beats the anchors is the flat one:

| | BACE | BBBP | |
|---|---:|---:|---|
| **our flat, 1B** | **83.6** | **70.5** | |
| **our graph, 1B** | 81.8 | 67.2 | |
| Llama-2-7B-chat LoRA | 74.8 | 65.6 | **gate B3 — we beat it on both, at 1/7 the parameters** |
| InstructMol-G, 7B + graph tokens | 84.3 ±0.6 | 68.6 ±0.3 | **gate B2 — both our arms inside the 3-point band; our flat BBBP exceeds it** |
| Uni-Mol (3D, ~19M-molecule pretrain) | 85.7 | 72.9 | the ceiling, not the target (§2.2) |

A 1B model with no molecular pretraining landing 0.7 below InstructMol-G on BACE and 1.9 *above* it
on BBBP is a real result about the *backbone plus adapters*, and it is worth stating separately from
the graph-vs-flat question, because it is the part that does not depend on GTLM at all.

**Three things this table is not.**

1. **Not a fully-tuned comparison.** §3.2.5 disclosure 2 carries over unchanged: `bias_lr` is
   graph-arm-only, so this is a tuned arm losing to an *untuned* one. That makes the negative
   stronger, not weaker — but "our flat twin beats published 7B baselines" is a claim about an
   untuned configuration and should not be quoted as a tuned one.
2. **Not three splits.** All three seeds share one deterministic scaffold split, so the ± is
   training variance only and carries **no split variance at all**. The published rows we compare
   against are means over 3 scaffold splits. See §8.3 — val and test here are different populations,
   which also makes checkpoint selection on BBBP close to arbitrary.
3. **Not the whole gate.** B1 is written over three sets; HIV (M3b, `013`) has not run.

**What it does not settle, and must not be read as settling.** Whether the graph arm loses because
the *endpoints* are motif-driven (§3.2.5's mechanism, supported here) or because the Levi encoding
is the wrong graph for chemistry (§3.2's open question, which M4 exists to answer). Those predict
the same Tier-B result and are separated by the encoding sweep, not by more Tier-B runs.

---

## 3. Encoding — the decision this document exists to make

### 3.1 Default: `atom_levi_rich`

**Nodes = atoms. Node text = a short natural-language atom descriptor, never the bare symbol.**

```
"carbon aromatic ring deg3 H1"        rather than       "C"
"oxygen carbonyl deg1 H0"                               "O"
"nitrogen ring deg3 H0 charge+1"                        "N"
```

Three independent reasons:

1. **Fairness.** It is exactly the information a GNN receives in its atom feature vector (element,
   degree, formal charge, hybridization, aromaticity, ring membership, implicit H). A bare-symbol
   encoding would handicap us relative to every baseline in §2.
2. **Cost.** `CLAUDE_CONTEXT.md` §4.5 measures overhead as a function of nodes-per-token: 7.9× at
   1.0 and 1.45× at 0.031. Bare atom symbols are the worst corner of that curve.
3. **It is what makes this GTLM and not a graph transformer.** The backbone can read "aromatic",
   "carbonyl", "chiral" as language it already knows. Bare symbols reduce the LLM to an expensive
   random feature map.

**Bonds = Levi nodes** (D1 is decided: Levi). Text `"single bond"` / `"double bond"` /
`"triple bond"` / `"aromatic bond"`, `+ " in ring"`, `+ E/Z` where defined.

### 3.1.1 Measured at M0 (2026-08-28)

`analyse_dataset.py`, all nine Tier-B datasets, Llama-3.2-1B tokenizer, 800-molecule samples for
token counts, full corpus for sizes and splits.

| dataset | mols | dropped | atoms mean/p95/max | Levi N mean/p95 | flat SMILES tok | `rich_levi` tok | `terse_levi` tok |
|---|---:|---:|---|---|---:|---:|---:|
| `bace` | 1513 | 0 | 34.1 / 48 / 97 | 72.9 / 101 | 42 | 368 | 113 |
| `bbbp` | 2039 | 11 | 24.2 / 39 / 101 | 52.3 / 84 | 32 | 266 | 82 |
| `hiv` | 41119 | 8 | 25.6 / 47 / 116 | 55.2 / 98 | 30 | 281 | 90 |
| `tox21` | 7823 | 8 | 18.7 / 38 / 124 | 40.2 / 81 | 24 | 201 | 67 |
| `clintox` | 1478 | 6 | 26.2 / 55 / 136 | 56.0 / 117 | 39 | 283 | 87 |
| `sider` | 1427 | 0 | 33.6 / 82 / **492** | 71.0 / 169 | 46 | 342 | 106 |
| `esol` | 1128 | 0 | 13.3 / 26 / 55 | 29.0 / 56 | 14 | 148 | 53 |
| `freesolv` | 642 | 0 | 8.7 / 18 / 24 | 19.1 / 39 | 9 | 98 | 40 |
| `lipo` | 4200 | 0 | 26.9 / 38 / 65 | 58.1 / 82 | 31 | 305 | 96 |

Three corrections this forced on the plan:

1. **The size estimate held; the cost claim was overstated.** Measured nodes-per-token is **0.20
   rich vs 0.61 terse — a 3× reduction, not the 6× §3.1 claimed** — because Levi bond nodes carry
   short text in both arms and dilute the effect. Both arms sit in the interior of the overhead
   curve; neither is in the bad corner.
2. **SIDER has a 492-atom molecule** (~1000 Levi nodes). `max_nodes` cannot be set from the mean,
   and SIDER alone decides whether the D5 cap needs a policy rather than a number.
3. **One bond type cannot be encoded, and it is negligible.** Across 1.63M bonds: 1,625,615
   single/double/triple/aromatic and **10 dative**, all in organometallic iron complexes. Dative
   bonds are directional and the Levi encoding is undirected, so those 3 molecules are dropped and
   counted (`is_encodable`) rather than silently mis-encoded. Parse failures counted separately: 30
   across the corpus, mostly BBBP's known-bad SMILES.

**Addendum 2026-08-31 — the flat arm is not as cheap as this table says on atom-named families.**
The `flat SMILES` column above is the *unlabelled* string. `atom_labels=True` (`data.py:405`) sets
an atom-map number on **every** atom, so the string the flat arm actually reads on an atom-named
family is `[O:1]1[CH2:2][CH2:3]...`, measured over 300 molecules each:

| | plain SMILES | with atom maps | vs the `rich_levi` prefix |
|---|---:|---:|---|
| `bace` | 47.5 | **181.8** | 368 → +13% plain, **+49% labelled** |
| `bbbp` | 31.7 | **120.0** | 266 → +12% plain, **+45% labelled** |

A 3.8× blowup. It changes no result — both arms were always measured as configured — but it does
mean the flat twin's prompt on `ring_membership`/`ring_size`/`fg_atom_membership`/`stereo_assigned`
is ~4× the length §3.1.1 implies, and any future cost argument that quotes "SMILES is 42 tokens"
is quoting the wrong number for half the suite. The molecule-level families
(`fg_count`, `fg_presence`, `ring_count`, `longest_chain`, `stereo_potential`) name no atom and do
use the plain string.

### 3.1.2 Cost — measured at M2 (`000_smoke`, job 134071)

Six runs, 30 steps, `ring_membership` on BACE, batch 4 × accum 8. **No accuracy from this sweep is
quotable**; speed and memory are what it was for.

| arm | bias | impl | ms/step | peak GB |
|---|---|---|---:|---:|
| flat | none | eager | **451** | 11.5 |
| graph | none | eager | **637** | 26.1 |
| graph | spd+magnetic | eager | 2177 | 26.1 |
| graph | spd+magnetic | flex | **1493** | **20.8** |

**Flex vs eager is settled, and the answer is conditional.** With the bias **on**, flex is 1.46×
faster and uses 20% less memory, and it halves the bias overhead (eager 3.4×; flex 1.7×). With the
bias **off**, eager wins on both arms — there is nothing for flex's block-skipping to skip at
N ≈ 59 (measured token sparsity 0.009). **Decision: flex for the graph arm, eager for the
controls.** At 30 steps the ~320 s autotune is not amortised, so `flex_compile_mode` matters more
for short runs than this ranking suggests.

**Cost of the whole architecture on molecules: 3.3×** plain-LLM using each arm's best backend
(4.8× eager-to-eager). In line with the ~3× published figure. **47 ms per example** on the best
graph configuration. Molecules are the cheapest domain in the mixture.

### 3.2 Encoding arms

**Two axes, easy to conflate.** Axis 1 is *how much text a node carries*; axis 2 is *whether a bond
is a node at all*.

| | `levi` — bond becomes its own node | `atom_only` — no bond nodes |
|---|---|---|
| **`rich`** — `"carbon aromatic ring deg3 H1"` | **the default** | bond order folded into the atom's text (`"deg3 arom2 single1"`) |
| **`terse`** — `"carbon"`, `"double"` | the cheap honest baseline — **run it** | **✗ never run.** Information-destroying: with no bond node *and* no bond summary, a double bond is indistinguishable from a single one. Not a weak arm — an invalid one. |

**Three cells, not four**, the fourth discarded by construction. `terse × levi` matters most for
§3.1's argument: if it matches `rich × levi`, the featurizer is unnecessary — a cheap result either
way, and worth knowing before M4 spends anything.

**Pre-registered prediction (2026-08-29, before M4 runs): `rich × levi` wins, and its margin
grows with task difficulty.** Two things falsify it: `terse × levi` matching `rich × levi` on the
hard families (⇒ the featurizer is dead weight), or the `rich` margin *shrinking* as families get
harder (⇒ rich text helps the easy lookups where the answer is nearly copied out of a node's own
description — the opposite of the stated mechanism, and a more interesting result than
confirmation). §3.2.1's asymmetry cuts for the prediction on ring/aromatic families specifically,
so check whether any `rich` win is confined to those rather than being the general effect.

**`atom_only` is not a formality.** Levi doubles N, doubles every shortest path, and turns a
benzene 6-ring into a 12-cycle — and the `substructure` probe says the *spectrum* carries ring
detection. If `atom_only` wins on Tier A, that is a finding for D1: Levi is right for *text-bearing*
edges (KGQA relations carry sentences) and possibly wrong for *typed* edges with a four-symbol
vocabulary.

**`+smiles` is an arm, never the default.** InstructMol-GS vs -G is +3.8 on BBBP and −5.1 on HIV —
genuinely task-dependent. The headline must be `graph_only`, or the claim is unclean in exactly the
way the WebQSP triplet arm is (`project-triplet-excluded-from-gtlm-claims`). A SMILES string in the
prompt is a flat serialization sitting inside a graph arm.

**Atom referencing (`atom_labels`).** Tier-A questions name an atom, so both arms name one:
`atom14` prefixed to the node's text, `[cH:14]` atom-map numbers in the flat SMILES. Labels are
**1-based in both**, because RDKit reads map number 0 as "unmapped" and a 0-based scheme drops atom
0's label *in the flat arm only* — silently, and in the direction that flatters the graph arm.
Pinned by a test.

### 3.2.1 What each encoding preserves — measured at M1

`roundtrip_check` encodes a molecule, rebuilds one **from node text and topology alone**, and
compares. Over every molecule in Tier B, all three encodings — **61,369 × 3 round trips, zero
failures.** Four iterations to get there, each failure a real finding:

| What failed | Cause | Resolution |
|---|---|---|
| `rich_levi`, ~1% | isotopes (`[99Tc]`) and radical electrons (`[N]=O`) were **not in the node text** | added `iso<n>` / `rad<n>` — a real gap in the rich-feature spec |
| `rich_levi`, 1 molecule | an explicit `[H]` atom alongside its parent's H count, double-counted | `RemoveAllHs` at load. Not `RemoveHs`, which keeps an H that defines bond stereo — exactly this molecule |
| `terse_levi`, 6.3% | comparing as SMILES asks terse for something it cannot give | comparison level corrected to `labelled_graph` |
| `terse_levi`, the rest | `SanitizeMol` **kekulises**, rewriting as single/double the aromatic bonds the encoding carried correctly | compare the unsanitised rebuild; the mutation was in the decoder |

**The terse finding is substantive and is a prior for the `terse × levi` arm.** Aromaticity
perception requires hydrogen counts: pyrrole's `[nH]` is aromatic *because* that nitrogen carries
an H. `terse` drops the count, so the ring cannot be re-perceived. The labelled graph survives
intact, but **the molecule is no longer reconstructible as a chemical object** for 6.3% of
drug-like molecules, concentrated in N-heterocycles. A `terse` loss should be attributed to *this*
and checked against the aromatic subset before being read as evidence about node-text richness.

### 3.2.2 What the M2 campaign settled, in brief

Ten sweeps, ~90 runs, ~50 GPU-h. The full narrative is in git history; the durable outcomes:

| Question | Answer | Where |
|---|---|---|
| Does the graph arm learn at all? | Yes. The 0.000 at 30 steps was undertraining; the unfixed baseline reaches 1.000 at 1250 steps. Bias parameters verifiably leave init (6.27 → 23.9). | 3.2.3 |
| Is `node_position_mode` needed? | No. **Unwired entirely** — not defaulted off, removed. | 3.2.3 |
| Is the question node load-bearing? | Yes, and it stays **on**. | 3.2.3 |
| Which families can discriminate? | Four of ten saturate in both arms and can rank nothing. | 3.2.4 |
| Is `max_spd = 32` the limitation? | **No, measured not assumed.** Levi diameter p50 26 / p90 37 / max 76; 26–30% of examples have a pair at or beyond 32, but the mean share of *pairs* clamped is 0.66–0.84%. If the clamp bound accuracy there would be a step at 32; there is none — on `longest_chain` accuracy *rises* across it (d≥22 0.734, d≥28 0.794, d≥32 0.826). **`max_spd` stays 32.** Raising it is cheap, which is precisely why it needed a reason. | 3.2.5 |
| Does the prompt hub shortcut the molecule? | No. Prompt edges are **directed**: out-degree 13, in-degree 0, and removing the node changes **zero** atom-atom shortest paths. (The first diagnostic called `to_undirected()` before measuring and answered a question about a different graph.) | 3.2.5 |
| Was the training recipe a confound? | **Yes, and it was the single biggest one.** | 3.2.7 |
| Is the structural bias load-bearing? | Yes where the task is topological, and it is **SPD**, not magnetic. | 3.2.8 |
| Does the graph arm beat the flat twin on Tier A? | **4 wins, 1 tie, 2 losses.** | 3.2.5 |

### 3.2.3 Settled: question node on, `node_position_mode` gone

The M2 canary ran two candidate fixes against the unfixed baseline at 1250 steps on
`ring_membership`: graph/`reset` **1.000**, graph/`spd_depth` 0.998, flat 0.877. **The unfixed
baseline solves the task perfectly; neither fix was needed.**

**DECISION 2026-08-29: `node_position_mode` is unwired from this experiment entirely.** No
`RunConfig` field, no flag, no run-record entry; `train.py` is back on the shared
`expressiveness/training/dispatch.build_collator`. A test asserts the field cannot be set. The
reasoning is about evidence: `spd_depth` has **two measurements and no positive result** — kgqa E3
measured **−9.4 F1** on WebQSP (0.7351 → 0.6412, 3 seeds) and rejected it, and the canary put it
0.002 lower here with less bias movement. A knob that is wired but never set is worse than no knob:
a sweep can put it in the axis list and the run record reports it faithfully, producing a campaign
arm that was never justified. `GraphCollatorV2` still implements it and kgqa owns it as a
documented negative result — that is the right home.

**The RoPE-shock mechanism behind it is real but is not a defect.** An untrained model does
continue the atom description (top-5 at the answer position is `' ring'`, `' deg'`, `' carbon'`
for the graph arm vs `' True'`, `' yes'`, `' Yes'` for flat), and under `reset` the prompt node's
`"\nA:"` does sit at RoPE positions 0–2, inside the prefix's own range. 1250 steps of LoRA
overrides that prior completely. Read it as a description of the initialisation, not a bug report —
and as a caution that a measured mechanism is not by itself a measured *problem*.

**`question_node` stays, and stays on.** The question lives in the **prefix**, in its own edge-free
node, so every atom and bond node attends to it and node representations are question-conditioned.
Edge-free is the point — the question is visible through attention but contributes nothing to the
SPD / magnetic features, so the structural bias still describes the molecule alone.

**Renamed 2026-08-29: the values are `"on"` / `"off"`, not `"isolated"` / `"off"`.** graphqa
and kgqa keep `"isolated"` because there the placement is one of several conceivable ones; here
only one is worth having. `"isolated"` is **rejected**, not aliased — a silently-accepted synonym
would put two spellings of one arm into the run records. `002_difficulty`'s `runs.jsonl` straddles
the rename; group the two spellings together. The dataset cache is unaffected (`dataset_path` tags
the key only when the value differs from the default; pinned by a test).

### 3.2.4 Family selection, and the criteria that failed

Admission criteria for entering the M4 encoding sweep, fixed before results landed:

1. **Headroom** — graph accuracy ≤ 0.95. At ceiling nothing can be ranked.
2. **Signal** — graph − flat ≥ 0.03. Below that the family measures the LLM's handling of SMILES
   text, not the graph channel.
3. **Learnability** — graph accuracy well above the majority-class rate.

**Applied literally to the 2500-step screen, these admit one family** — because criterion 2 was
written on the assumption that where there is headroom the graph arm leads, and at the stale recipe
it did not. **Do not weaken the criterion to fit the result.** The right reading was that the
screen had found something the plan did not predict, and §3.2.7 is the diagnosis. Post-recipe-fix
(§3.2.5) the criteria are usable again; four families saturate in both arms and are excluded by
criterion 1 alone.

#### 3.2.4.1 The confounds this screen found — enforced in code, not remembered

**Accuracy at the first eval is not a measurement.** `fg_count`'s answer distribution is 76.0%
`" 0"`; both arms scored ~0.74 at eval 1, i.e. **both were predicting the mode and had learned
nothing**. Concluding anything from that compares two constants.

**Start-up cost is not a lower ceiling.** On `fg_count` the graph arm sat at base rate for 500
steps while flat departed it by step 200, then climbed steeply and was **still climbing when the
cosine schedule annealed the LR to min**. The model is being handed a modality it has never seen,
where flat SMILES is something Llama already read in pretraining.

Three consequences, all now in code:

1. **`base_rate`, `answer_distribution` and `n_classes` in every run record** (`train.py::_answer_stats`,
   read from the `.gtds` sidecar so a warm cache still reports it). Criterion 3 was unverifiable
   without it.
2. **`eval_curve`, `tail_gain` and `still_improving` in every run record** (`train.py::_convergence`).
   An arm gaining >1pp over the last three evals is **budget-limited** and its headline is a lower
   bound; the log now says so in words. Comparing a converged arm against an interrupted one is not
   a comparison.
3. **Judge on validation curves plus convergence state, never a single test scalar at one seed.**

**When budgets are mixed, mark them** — every cell at a longer budget carries a `*` and the table
says so. Two rules behind it, both still binding: **re-run the pair, not the cell** (otherwise a
starred comparison pits 2500 steps against 1250, replacing a bias against the graph arm with one in
its favour), and **trigger on the convergence flag, not the score** (re-running only the cells that
lost is selecting on the outcome). No current table mixes budgets; §3.2.5's is uniformly 5000 steps.

**`fg_count` and `fg_presence` are weak instruments by construction:** a 0.760 base rate leaves a
24-point usable range. `stereo_potential` (0.285) is much better shaped. Base rate belongs in the
admission decision alongside headroom and signal.

### 3.2.5 M2 CLOSING TABLE — the Tier-A result (`014`, job 135331, 20/20 COMPLETED, 2026-09-01)

Twenty runs, all COMPLETED: ten families × both arms, one recipe
(`lr 3e-5, lora_r 16, 5000 steps`; `bias_lr` 1e-2 on the graph arm as `008` ran it, 5e-3 on the flat
arm where it is inert), one dataset build, **molecule- and scaffold-disjoint splits** (§3.2.10.1).
Exact-match accuracy on 1000 test examples; σ is the unpooled two-proportion sampling bound.

| family | base | **graph** | **flat** | Δ | σ | verdict |
|---|---:|---:|---:|---:|---:|---|
| `longest_chain` | 0.500 | **0.988** | 0.828 | **+0.160** | +12.9 | **graph** |
| `ring_size` | 0.503 | **0.978** | 0.849 | **+0.129** | +10.5 | **graph** |
| `stereo_potential` | 0.527 | **0.767** | 0.724 | **+0.043** | +2.2 | **graph** |
| `ring_membership` | 0.500 | **1.000** | 0.959 | **+0.041** | +6.5 | **graph** |
| `fg_atom_membership` | 0.531 | **0.979** | 0.941 | **+0.038** | +4.4 | **graph** |
| `aromatic_ring` | 0.504 | 1.000 | 0.998 | +0.002 | +1.4 | tie (both at ceiling) |
| `fg_presence` | 0.849 | 0.938 | 0.946 | −0.008 | −0.8 | tie |
| `fg_count` | 0.849 | 0.862 | **0.909** | **−0.047** | −3.3 | flat |
| `ring_count` | 0.294 | 0.862 | **0.939** | **−0.077** | −5.8 | flat |
| `stereo_assigned` | 1.000 | 1.000 | 1.000 | — | — | **VOID** — single-answer test split |

**5 graph wins, 2 ties, 2 flat wins, 1 void.** Mean Δ +0.031, median +0.038.

**`stereo_assigned` is void, not a tie.** All 1000 of its test examples carry the same answer, so a
constant predictor scores 1.000 and the arms are formally incomparable. Recorded as
`degenerate_test_split` in the run record and voided by the report rather than quoted; §3.2.10.1 has
the cause (scaffold-splitting plus an HIV-dominated pool). It reported **1.000 for both arms**,
which is exactly the kind of number that would otherwise be read as a triumph.

#### What the re-run changed, family by family

Every number below is the same experiment measured twice: once on splits where up to 73.6% of the
test set was a memorised training item, once on molecule-disjoint splits.

| family | old Δ | old verdict | **new Δ** | **new verdict** | |
|---|---:|---|---:|---|---|
| `longest_chain` | +0.044 | graph | **+0.160** | graph | margin **3.6×** |
| `ring_size` | +0.115 | graph | **+0.129** | graph | grew |
| `stereo_potential` | +0.025 | graph | **+0.043** | graph | grew |
| `ring_membership` | +0.020 | graph | **+0.041** | graph | grew |
| `fg_atom_membership` | +0.012 | tie | **+0.038** | **graph** | tie → win |
| `fg_count` | −0.048 | flat | −0.047 | flat | reproduced exactly |
| `fg_presence` | −0.044 | flat | −0.008 | **tie** | flat win → tie |
| `aromatic_ring` | +0.003 | tie @ ceiling | +0.002 | tie | unchanged |
| `ring_count` | 0.000 | tie @ ceiling* | **−0.077** | **flat** | new result |
| `stereo_assigned` | +0.001 | tie @ ceiling* | — | **void** | family lost |

<sub>* quoted in the old table from `004_extended` at a different recipe, never run at this one.</sub>

**THE HEADLINE SURVIVED AND STRENGTHENED, and the direction of the correction is the surprise.**
Every one of the four graph wins **grew**; none shrank. A fifth family crossed from tie to win. The
contamination had been *understating* the graph arm, because the arm exploiting the memorisable test
items was the **flat** one — on `longest_chain` the flat arm falls 0.947 → 0.828 while the graph arm
holds 0.991 → 0.988. That is the opposite of what §3.2.10 feared when the defect was found, and it
is worth stating plainly: the damage assessment predicted margins would "move in both directions",
and on a clean measurement they moved almost entirely one way.

Absolute accuracies are lower nearly everywhere, as they must be once memorisation is removed and
the pool widens (§3.2.10.1 has the base-rate shifts). **Read Δ, not the level**, when comparing with
anything before 2026-09-01.

#### The line is representational explicitness, not topology

§3.2.5's previous reading — "the graph arm wins topology and loses chemical-motif recognition" — no
longer covers the data, and the family that breaks it is new:

* **`ring_size` +0.129 (graph) against `ring_count` −0.077 (flat).** Same rings, same molecules,
  opposite verdicts. Ring count is as topological as ring size, so "topology" cannot separate them.
* **What does separate them is what SMILES writes down.** A SMILES string marks every ring closure
  with an explicit paired digit, so *counting* rings is a **lexical** operation on the string —
  count the digit pairs — needing no topology at all. Ring **size** is the number of atoms between
  a matched pair, which requires traversing the string as a structure; ring **membership** requires
  deciding whether a named atom lies inside such a span. The flat arm wins precisely the question
  its representation answers by lookup, and loses the two that require traversal of the same rings.
* **The motif families fit the same rule.** A functional group is a short substring — `N(=O)=O` for
  nitro — so `fg_count` and `fg_presence` are substring matching, and the flat arm keeps them. The
  Levi transform scatters that substring across nodes and edges, which is why `fg_count` is the
  graph arm's worst family (0.086 skill above floor).
* **And it explains the one that moved.** `fg_atom_membership` is the motif question re-asked about
  a *named atom* — that anchor is not lexically adjacent in the string, so the lookup shortcut
  breaks and the graph arm now **wins** it (+0.038) where it previously tied.

The rule that covers all ten families: **the flat arm wins what SMILES makes explicit as text; the
graph arm wins what has to be computed from structure.** This is a stronger claim than "topology"
because it predicts the `ring_size` / `ring_count` split in advance rather than absorbing it, and it
is falsifiable. Two standing predictions, neither yet measured: the graph arm should **win**
`bond_path` (a shortest path between two named atoms is traversal with no lexical shortcut —
held out under §4.1, so this is a prediction about a family we have deliberately never run), and a
hypothetical "how many bonds does this molecule have" family should go to the **flat** arm, since
SMILES writes every bond explicitly. **Treat it as the leading hypothesis, not a settled result:
it was formed after seeing `ring_count`, and §0's whole point is that a story fitted to data is not
evidence.** The encoding sweep (M4) is where it gets tested properly.

#### Disclosures that belong beside this table wherever it is quoted

1. **One seed.** Every σ is a two-proportion sampling bound at n = 1000 and says **nothing** about
   seed-to-seed training variance. M3a measured that variance on Tier B at 0.013–0.021 ROC-AUC, so
   it is not negligible. The marginal cells here — `stereo_potential` +2.2σ, `aromatic_ring` +1.4σ,
   `fg_presence` −0.8σ — are the ones a second seed could move. The four wins above +4σ are not.
2. **The arms are not equally tuned.** `bias_lr` is graph-arm-only by construction, so this is a
   tuned arm against an untuned one. The flat arm keeps `lr 3e-5` inherited from graphqa, and 006
   measured it responding to `lr` (+0.029 on `fg_count`). A flat `lr` sweep is what would earn the
   claim "beats a fully-tuned baseline"; **until then, do not make that claim.**
3. **Two families carry almost no signal.** `fg_count` and `fg_presence` sit at a 0.849 base rate
   after the pool widened, leaving 0.151 of headroom — `fg_count`'s graph arm clears its floor by
   1.3 points. Their margins are bounded by headroom rather than by the arms, and neither should be
   quoted as evidence of much in either direction. `aromatic_ring` is at the ceiling in both arms
   and is equally uninformative.
4. **Convergence is confirmed, not assumed.** All 20 runs peak between 0.00 and 0.79 of their
   budget with none still improving (recomputed per §8.2, since 2 of the 20 records carry the old
   inverted flag). These are ceilings, not interruptions.

**What this predicted about Tier B, and how that turned out.** The previous table's prediction —
recorded before M3 ran — was that Tier B's motif-driven endpoints put the graph arm on the losing
side of its own split. **M3a confirmed it** (§2.6: gate B1 fails on both BACE and BBBP). The
explicitness reading above says the same thing with a sharper mechanism: SMILES is a *good*
representation for pharmacophore recognition, so a graph channel has to earn its place on tasks that
require traversal, and property prediction largely does not.

### 3.2.6 RETRACTED: `005_fgcount_ablation`'s "the bias is inert" null

The sweep reported `spd+magnetic` 0.820 / `spd` 0.822 / `none` 0.823 on `fg_count`, read at the
time as the bias contributing nothing. **That reading is retracted: the sweep ran at
`bias_lr = 1e-3`, the lowest in the repo** (§3.2.7). `bias_lr` is *the magnitude knob*
(`project-landmark-campaign`), so "the bias does nothing" and "the bias never reached useful
magnitude" are indistinguishable here. **Do not cite the 0.820 / 0.822 / 0.823 null.** §3.2.8
re-measured it properly and the conclusion happens to survive — on stronger evidence, for a
different reason. The question-node arm survives as a direction (`off` is worse, −0.022) but its
magnitude was measured at the stale recipe and should not be quoted precisely.

### 3.2.7 `006_recipe` — the recipe was the blocker

Molecules inherited `lr=1e-5, bias_lr=1e-3, lora_r=8` from `expressiveness`, the oldest experiment
in the repo, and never revisited it. Measured across every campaign's run records that is the
**lowest-capacity recipe anywhere**:

```
relbench          lr 2e-4   bias_lr 5e-2   r 32
kgqa / context    lr 1e-4   bias_lr 5e-3   r 64
probes            lr 5e-5   bias_lr 5e-3   r 16
graphqa           lr 3e-5   bias_lr 5e-3   r 16     (329 runs)
molecules         lr 1e-5   bias_lr 1e-3   r 8      <-- 3-20x / 5-50x below
```

Eight runs, 5000 steps: two tasks × two arms × `current` and `tuned` (graphqa's — the closest
comparable and the one with the most runs behind it, rather than newly invented values). **Both
arms get the identical change**; tuning only the graph arm would replace a bias against GTLM with
one in its favour. Running `current` at 5000 too separates *more steps* from *more capacity*.

| family | arm | 1250 | 2500 | 5000 `current` | 5000 `tuned` |
|---|---|---:|---:|---:|---:|
| `longest_chain` (base 0.253) | flat | 0.844 | 0.934 | 0.948 | 0.947 |
| | **graph** | 0.778 | 0.855 | **0.959** | **0.989** |
| | *gap* | *−0.066* | *−0.079* | *+0.011* | ***+0.042*** |
| `fg_count` (base 0.760) | flat | 0.848 | 0.907 | 0.912 | 0.936 |
| | **graph** | 0.809 | 0.820 | 0.841 | 0.897 |
| | *gap* | *−0.039* | *−0.087* | *−0.071* | *−0.039* |

1. **`longest_chain` flips sign** — the first molecule-level win in the domain, with no change to
   encoding, wiring or task.
2. **The recipe change is graph-specific on that task**: it moved flat by −0.001 and graph by
   +0.030. `bias_lr` and adapter rank are the only knobs touching the graph arm alone, which
   directly implicates §3.2.6's null.
3. **`still_improving` under-called undertraining.** The detector marked `longest_chain` graph
   **converged** at 2500 steps (0.855); it then gained **+0.104** at the same settings. §3.2.4.1
   installed that flag to stop exactly this error. **Treat `still_improving = False` as weak
   evidence, never as licence to conclude a plateau.**
4. **`fg_count` is a genuinely different case.** The gap halves but does not close, and both
   `current` arms are still improving at 5000 steps — those cells are lower bounds.

**Consequences.** **The tuned recipe is the recipe**: M3, M4 and everything after run at
`lr=3e-5, bias_lr=5e-3, lora_r=16`. Every pre-`006` number is a lower bound for the graph arm —
cite it as such or re-measure.

### 3.2.8 `007_tuned_ablation` — the win is the bias, and the bias is SPD

§3.2.7 moved three knobs at once, admitting two readings: **(1)** `bias_lr` finally let the
structural channel reach useful magnitude, or **(2)** `lora_r 8 → 16` plus 3× the lr gave the
*adapter* enough capacity to read path structure out of the node text, bias still inert. Reading 2
is not a strawman — it is what §3.2.6 measured. Four runs at the tuned recipe, 5000 steps,
identical to 006's tuned graph cells except the bias channel:

| family | `none` | `spd` | `spd+magnetic` *(006)* | flat *(006)* |
|---|---:|---:|---:|---:|
| `longest_chain` (base 0.253) | 0.938 | **0.983** | **0.989** | 0.947 |
| `fg_count` (base 0.760) | 0.888 | 0.888 | 0.897 | 0.936 |

`spd` − `none`: **+0.045 (5.2σ)** on `longest_chain`, +0.000 (0.0σ) on `fg_count`.
`spd+magnetic` − `spd`: +0.006 (1.1σ) and +0.009 (0.7σ).

1. **Reading 2 is dead.** Strip the bias and the graph arm scores 0.938 at `lora_r=16, lr=3e-5` —
   statistically indistinguishable from the flat twin (0.947, 0.9σ). Adapter capacity buys parity
   with SMILES; **the bias buys the win.**
2. **It is SPD; magnetic adds nothing measurable.** `spd` alone reaches 0.983 of the 0.989. **This
   contradicts §3.3's expectation** that magnetic should be load-bearing because cycle detection is
   where spectral features classically win (the probe measured 97.9 vs 90.8). §3.3's claim is now
   *tested and unsupported*, not untested.
3. **`fg_count`'s null is real and properly warranted this time.** `none` and `spd` are identical
   at 0.888, and critically `bias_norm_final = 40.40` against `bias_norm_init = 0.0` — **the bias
   module trained to substantial magnitude and still contributed nothing**, which is what
   `feedback-verify-nulls-are-real` demands and what §3.2.6 could not show. 006's `fg_count` gain
   (0.841 → 0.897) was adapter capacity.
4. **The bias is a large sample-efficiency win, not only an accuracy win.** `spd` reaches 0.950 val
   at **step 800**; `none` needs the full 5000 to reach 0.944–0.950. ~5.5× fewer steps, and the
   `spd` eval loss plateaus at 0.137 against `none`'s 0.51.
5. **The caveat that keeps this honest.** Both families are molecule-level, so the split is not
   local-vs-global. This section proposed *"the bias is load-bearing where the task reduces to
   graph distances"* — **§3.2.5 falsified that** and replaced it with topology-vs-motif. What
   survives is the ablation itself: on `longest_chain` the structural channel, not the adapter,
   carries the win.

**CLOSED 2026-08-31 (`011`).** This section long read "every cell here is one seed; a second seed on
the four `longest_chain` cells is the last thing standing between this and a claim fit for
`src/generalist/PLAN.md`." That seed is now run. Every cell moves by ≤0.023 between seeds and the
ordering is identical, so the ablation is **not** a one-seed artifact. See §3.2.10 for the two-seed
table — and read it there rather than here, because these cells are memorisation-inflated and the
honest version of this result is the novel-molecule one.

### 3.2.10 INCIDENT, CLOSED: Tier A's test split was not molecule-disjoint (found 2026-08-31, fixed and re-measured 2026-09-01)

> **Status: resolved.** The generator was fixed the same day (§3.2.10.1) and every affected number
> was re-measured on molecule-disjoint splits by `014`; §3.2.5 carries the replacement table and the
> conclusions held. This section is kept as the incident record — what broke, how it was caught, and
> what the post-hoc repair got wrong — not as a live caveat.

**How it was found.** Gate A2's negative control ran for the first time (`011`) and *fired*:
`stereo_assigned` with `stereo_tags: off` scored **0.936** against a 0.715 base rate, where §1
predicts chance, because chirality reaches the graph only through the parity tag. §2.5 gate A2
says to find the leak before reading any other row. There is no leak in the encoder —
`stereo_tags` gates the atom parity word (`data.py:153`) *and* bond E/Z (`data.py:207`), so with
it off the graph carries no stereochemical text. The cause is the data.

**The mechanism.** `generate_examples` draws `pool[rng.randrange(len(pool))]` — **with
replacement** — from 3552 molecules until it has 5500 examples, and `prepare_dataset` then slices
that list positionally into 4000/500/1000. Nothing makes one slice's molecules disjoint from
another's. For a **molecule-level** family the example is a deterministic function of the molecule,
so a molecule recurring across the boundary is an *exact* duplicate — same graph, same question,
same answer — and memorising it answers the test item.

Measured by replaying the generation at `data_seed 0` (the replay reproduces each cached dataset's
recorded answer distribution exactly, so it describes the real training data):

| family | level | test molecules seen in train | **exact duplicates** |
|---|---|---:|---:|
| `longest_chain` | molecule | 73.6% | **73.6%** |
| `ring_count` | molecule | 72.6% | **72.6%** |
| `stereo_potential` | molecule | 72.5% | **72.5%** |
| `stereo_assigned` | molecule | 72.4% | **72.4%** |
| `fg_presence` / `fg_count` | molecule | ~69% | ~11% *(the asked group varies)* |
| `ring_size`, `ring_membership`, `aromatic_ring`, `fg_atom_membership` | atom | ~70% | ~6% *(the named atom varies)* |

**Re-scored on unseen molecules — no GPU required.** `per_example/*.jsonl` already records `i` and
`correct` for every test item, and the generation is deterministic, so each row maps back to its
molecule. Every run's recomputed overall accuracy reproduces its recorded `test_accuracy` (asserted,
not assumed — a mismatch would mean the mapping is wrong and the numbers meaningless).

**Accuracy on the duplicate subset is 1.0000 in almost every run**, which is the mechanism
confirmed rather than inferred.

**SUPERSEDED 2026-09-01 — `014` re-measured all of this properly, and §3.2.5 now holds the result.**
A post-hoc re-scoring was done at the time (re-computing each run's accuracy on the subset of test
molecules never seen in training, using the stored `per_example/*.jsonl`), and it is no longer worth
carrying: it could not fix either of the two things that mattered. **Power** — it left ~270–316
examples where the design intended 1000. And **selection** — `load_best_model_at_end` had chosen the
checkpoint by val AUROC, and the *val* split overlapped train exactly as the test split did, so
which weights were kept was itself contaminated, and no subsetting can undo that.

Its conclusions are recorded here only as a matter of record, because the clean re-measurement
disagreed with one of them in an interesting way. The re-scoring found every verdict direction
surviving but magnitudes moving **both** ways — `ring_size` +0.115 → +0.103 and `ring_membership`
+0.020 → +0.013 both *shrank* — and concluded that the contamination had inflated some margins.
**`014` shows the opposite: on a clean measurement every graph margin grew** (§3.2.5). The
subsetting was misleading in the one direction it was supposed to be conservative about, which is a
reason to distrust post-hoc repair of a broken split in general and to re-run instead.

*(An earlier draft of this section additionally claimed "every margin widens", computed on a weak
novelty definition that excluded only exact `(molecule, question)` duplicates rather than every
molecule seen under any question. That claim was withdrawn at the time.)*

**§3.2.8's ablation, re-scored and now at two seeds** (`011` supplied seed 1). Note this table is
the *ablation*, not the headline: `none` is the graph arm with its structural channel deliberately
removed, and it is a control, not the arm the campaign reports.

| `longest_chain`, novel molecules | seed 0 | seed 1 |
|---|---:|---:|
| graph, `none` — **bias switched off, the control** | 0.7689 | 0.7576 |
| **flat twin** | 0.8030 | 0.8220 |
| graph, `spd` | 0.9356 | 0.9583 |
| **graph, `spd+magnetic` — the reported arm** | **0.9659** | **0.9697** |

The ordering `none < flat < spd < spd+magnetic` replicates at both seeds, and every cell moves by
≤0.023 between them — which also closes §3.2.8's stated hole ("every cell here is one seed"). The
`spd` − `none` gap is **+0.167 / +0.201** on novel molecules against the +0.045 originally reported.

This **strengthens** §3.2.8's finding 1 rather than qualifying it. On the memorisation-inflated
numbers, stripping the bias left the graph arm at *parity* with SMILES (0.938 vs 0.947, 0.9σ) —
"adapter capacity buys parity, the bias buys the win". On novel molecules it does not reach parity:
without the structural channel the graph arm is **behind** the flat twin by 3-6 points, and the
entire advantage over SMILES is attributable to the bias.

**What was compromised is the absolute number, not the comparison** — and `014` bore that out:
`longest_chain`'s graph arm moved 0.991 → 0.988 on a clean re-measurement while its flat twin fell
0.947 → 0.828. **Quote §3.2.5's post-fix numbers.** Nothing measured before 2026-09-01 should be
quoted as a Tier-A accuracy.

**Gate A2's off/on contrast — still the only measurement of it, and now unrepeatable on this pool.**
`014` widened the pool, which collapsed `stereo_assigned`'s test split to a single answer
(§3.2.10.1), so the family scores 1.000 in both arms and carries no signal. The numbers below are
therefore pre-fix and stand alone; re-running the contrast needs a pool where the family is not
degenerate (`bace,bbbp,tox21,lipo`). On novel molecules `stereo_assigned` goes 0.9928 with
the parity tag to **0.8152** without it — the off/on gap §1 asked for, and it is large. The control
does its job: the R/S information is genuinely not in a plain atom-bond graph. One caveat to carry:
0.8152 still sits ~10 points above the 0.715 base rate, and the parity channel cannot explain that.
A connectivity-only predictor is *not* the explanation — answering `n_potential` scores 0.500, and
assigned ≠ potential for 50% of the corpus. The most plausible remaining account is a **provenance
correlation**: whether a chemist specified stereochemistry at all correlates with molecular class,
which *is* in the graph. That is learnable from topology and is not a code defect, but it is
unverified and should be treated as an open question, not a settled one.

**Tier B is unaffected.** Its scaffold split is molecule-disjoint by construction and by test, so
M3a/M3b need no changes and the campaign's Tier-B numbers are clean. Everything above is about
Tier A alone.

#### 3.2.10.1 The fix, and the pool it forced

The generator now does what `tier_b.py` always did: **split the molecules first, generate examples
inside each split.**

* `split_molecule_pool` partitions the pool into molecule-disjoint train/val/test by
  **Bemis-Murcko scaffold** — not a random partition. Scaffold makes the test set *structurally*
  novel rather than merely unseen, which is the property a structural-reasoning claim needs, and it
  reuses the split Tier B already has under test. Pool fractions follow the requested example
  counts, so the whole sizing rule is "a single-example family needs a pool at least as large as
  the examples requested".
* `generate_examples` takes one split's molecules and consumes them **without replacement within a
  pass**, so no molecule is used twice until every usable one has been used once.
* `SINGLE_EXAMPLE_TASKS` (`longest_chain`, `ring_count`, `stereo_potential`, `stereo_assigned`)
  emit one example per molecule, so asking for more examples than the split has molecules **raises**
  rather than silently duplicating. Other families vary the named atom or the functional group and
  legitimately take further passes.
* The artifact path carries a **`molsplit`** tag, so the pre-fix `.gtds` files cannot be loaded by
  the fixed code — their paths simply do not match.
* `tests/experiments/molecules/test_tier_a_splits.py` pins all of it, including a test that
  reproduces the old one-pool draw and **requires the overlap to reappear** — a disjointness assert
  that cannot fail is decoration. Tier-A generation previously had **no test coverage at all**,
  which is why the defect survived a campaign.

**The fix is verified end to end, not just unit-tested (2026-09-01).** Running the real
`split_molecule_pool` + `generate_examples` at the sweeps' own sizes (4000/500/1000) across all ten
families: **zero** molecule overlap and **zero** Bemis–Murcko scaffold overlap between every pair of
splits, and **zero** test examples that also appear in train — against the 73.6% duplicate rate the
old sampler produced. The audit also found one thing the unit tests did not cover: a *multi*-example
family can re-emit a `(molecule, question)` pair on a later pass, because it picks the named atom at
random. At the four-corpus pool `fg_atom_membership` repeated 8 of 1000 test examples (871 distinct
molecules); every other family repeated none. That is a within-split repeat — it costs effective
sample size, it is not a train/test leak — but it is now **counted** into
`stats["repeats_by_split"]` and carried into the run record rather than left invisible.

**Pool: DECIDED 2026-09-01 (`014`).** The §3.2 sweeps set `pool: bace,bbbp` (3552 molecules), which
the fix makes *infeasible* rather than merely narrow: a proportional test pool is 646 molecules
against 1000 requested for the four single-example families, so those configs now **fail loudly**
instead of duplicating. Measured capacity, which is what settled it:

| pool | train / val / test molecules | verdict |
|---|---:|---|
| `bace,bbbp` | 2584 / 322 / 646 | infeasible — 646 < 1000 |
| `bace,bbbp,tox21,lipo` | ~11.3k / ~1.4k / ~2.8k | feasible; `fg_atom_membership` repeats 8/1000 |
| **all five** (`014`) | **41232 / 5154 / 10308** | 1000 distinct test molecules, 0 repeats, every family |

All five wins because it holds the **example counts** fixed at 4000/500/1000, so the re-run changes
the data and not the budget, and it preserves the n = 1000 test split — §3.2.10 already spent most
of this campaign's statistical power once, and shrinking the splits to save a pool label would spend
the rest. It is also `DEFAULT_POOL`, i.e. what `dataset.py` documents as intended; `bace,bbbp` was
the narrowing. **The cost is stated plainly and must travel with the new table:** HIV is 41k of the
56k molecules, so Tier A's chemistry is now HIV-screening-dominated, and absolute accuracies under
`014` are **not** comparable with §3.2.5's. The graph − flat margin *is*, because both arms draw the
same molecules, questions and answers in the same order — now pinned by
`test_both_arms_see_the_same_molecules_questions_and_answers`.

**MEASURED COST OF THAT CHOICE, and it is larger than `014`'s header predicted.** The distribution
shift was accepted in the abstract before its size was known. Measured from the built artifacts once
`014` began — majority-class rate of the **test** split, old pool → new:

| family | old base | new base | headroom (1 − base) |
|---|---:|---:|---|
| `stereo_assigned` | 0.715 | **1.000** | 0.285 → **0.000 — VOID** |
| `longest_chain` | 0.253 | 0.500 | 0.747 → 0.500 |
| `stereo_potential` | 0.285 | 0.527 | 0.715 → 0.473 |
| `fg_count` | 0.760 | 0.849 | 0.240 → **0.151** |
| `fg_presence` | 0.760 | 0.849 | 0.240 → **0.151** |
| `ring_count` | 0.326 | 0.294 | 0.674 → 0.706 |
| `ring_membership` | 0.505 | 0.500 | unchanged |
| `ring_size` | 0.498 | 0.503 | unchanged |

**`stereo_assigned` is destroyed outright**: all 1000 test examples carry the same answer, so a
constant predictor scores 1.000 and the family cannot compare two arms at all. Scaffold-splitting
sends the rarest scaffolds to test, and HIV's screening compounds — 73% of the pool — do not carry
assigned stereocentres. It is now flagged in the run record as `degenerate_test_split`, printed as a
loud warning by `train.py`, and voided by the report rather than quoted
(`tests/experiments/molecules/test_answer_stats.py`). Its loss costs little in substance — the old
table already had both arms at 0.997/0.996, "the designed-in loss, which never fired" — but it was
lost *silently*, and a family reporting a perfect score is the last number anyone audits.

**The three ring families are unaffected**, including `ring_size`, which carries the largest graph
win in the old table (+0.115). `longest_chain` and `stereo_potential` lose a third of their headroom
and keep plenty. `fg_count` and `fg_presence` are the badly compressed pair at 0.151 — and an early
`014` record has `fg_count` graph at **0.862 against a 0.849 floor**: 1.3 points of signal where the
old sweep had 12.8.

**In hindsight `bace,bbbp,tox21,lipo` was the better pool, and a future re-run should use it.** The
decision above traded "8 repeated test examples out of 1000 in one family" against "distribution
shift", and the shift is by far the larger cost: HIV at 73% of the pool dominates every family's
answer distribution, while the four-corpus pool is a balanced 15.5k — still far above the 1000
molecules a single-example family needs. `014` is **not** being cancelled over this. The families
carrying the load-bearing verdicts keep their headroom, both arms see identical data, and restarting
would spend real GPU-hours to re-measure the same comparison at modestly better resolution. But the
compression travels with the new table, and margins on `fg_count` / `fg_presence` are bounded by
headroom rather than by the arms.

**§3.2.5 is re-measured under this fix (`014`). §3.2.4 and §3.2.6–§3.2.9 are NOT** — those sections
report the difficulty screen, the recipe sweep and the bias ablation, none of which `014` re-ran, so
their absolute accuracies still come from contaminated splits. They are kept because their
*findings* are about the ordering of arms and recipes, which §3.2.5's clean re-measurement supports
rather than contradicts; but do not quote a number out of them as a Tier-A accuracy. The bias
ablation (§3.2.8) is the one worth re-running next — it is the decomposition behind the campaign's
central claim and it has never been measured on a clean split.

The tool that produced the post-hoc re-scoring (`duplicate_analysis.py`) has been **removed**: its
replay reproduced the old generator, which no longer exists, so keeping it would mean shipping code
that silently describes a dataset the repository can no longer build. It is recoverable from git
history (the commit adding it is titled *"molecules: re-score Tier A on unseen molecules"*).

**M4 is UNBLOCKED as of 2026-09-01.** It needed Tier-A numbers measured on molecule-disjoint splits;
`014` (job 135331, 20/20 COMPLETED) delivered them and §3.2.5 carries the table. The encoding sweep
can now be designed against real numbers — and §3.2.5's explicitness hypothesis gives it a sharper
job than "which of three cells wins": it predicts *which families* should separate the cells.

### 3.3 The bias channel behaves differently here

**Molecules are undirected, so the magnetic Laplacian's Hermitian phase is identically zero and
`magnetic` degenerates to plain-Laplacian spectral information** (`probes/README.md`, Probe 2).
Three consequences:

* The `direction` probe's headline (*magnetic is the only channel carrying edge direction*)
  **does not transfer**. On molecules magnetic is a spectral channel. Anyone reading a molecule
  ablation next to the KGQA one will assume otherwise unless it is written down.
* **`magnetic` was expected to be load-bearing here and is not** — §3.2.8 measured `spd` alone
  reaching 0.983 of `spd+magnetic`'s 0.989. Two families, one seed, so it is a weak refutation
  rather than a settled one, but the expectation is now tested.
* D2's cost proviso is nearly inert at N ≈ 52 (the per-layer/shared gap scales as O(N²·M·m)). **Do
  not import `G=4` from `bias_sharing` reflexively** — per-layer magnetic is affordable here.
* *Optional, probably a distraction:* orienting bonds by a canonical rule (CIP rank) would
  reactivate the magnetic phase. Note and drop unless Tier A shows a spectral ceiling.

### 3.4 Everything else follows D3

`question_node: on` (§3.2.3 — `"isolated"` is rejected here, not aliased), `k_hop = 0`,
`prompt_style: chat` with instruct weights, LoRA dropout 0.15, RCM ordering on, `max_spd = 32`
(§3.2.2), and from §3.2.7 onward **`lr = 3e-5`, `bias_lr = 5e-3`, `lora_r = 16`**. Flex for the
graph arm, eager for the controls (§3.1.2).

The question node carries the task instruction **and the endpoint name** — that is what makes a
multi-endpoint set one model rather than 39 heads, and it is the GTLM-natural choice: one example
per `(molecule, endpoint)` pair, endpoint named in text. (Not exercised by the scoped Tier B, all
three of whose sets are single-endpoint; it re-enters with Tox21/SIDER at M3c.)

---

## 4. Multi-task design — the part that feeds the trunk

Three arms. All at 1B. All on machinery that **already exists**: `graphqa/load_dataset.py` merges
`TextGraphDataset`s across tasks with a per-source `ds_label`; `kgqa/config.py` has
`train_datasets` / `eval_datasets` with per-dataset metric namespaces and a `selection_dataset`
knob.

| Arm | Training set | Question it answers |
|---|---|---|
| **1. Specialist** | one model per task | the per-task reference numbers, and the protocol-matched comparison to every §2 baseline; without these, arms 2–3 are uninterpretable |
| **2. Chemistry generalist** | all molecule tasks in one model, routed by the QUESTION node | within-domain interference and transfer — does `ring_membership` help BBBP? |
| **3. Cross-domain** | molecules folded into a Phase-1-style mixture (graphqa + probes + kgqa) | the admission-gate question: does chemistry cost the rest anything? |

Arm 2 is where the interesting result lives. The hypothesis worth pre-registering: **Tier A
transfers into Tier B.** Structural pretraining on free RDKit labels should improve scaffold-split
property prediction, because scaffold generalization *is* a structural-similarity problem. If it
does, that is a genuine result — a use for infinite free labels the corpus-bound baselines cannot
copy — and it is directly measurable as arm 2 minus arm 1 on Tier B.

**Scoping Tier B thins this arm** (§1): Tox21 and SIDER were the large within-domain multi-task
sets. With the primary three only, arm 2 is 3 property tasks + 9 Tier-A families rather than
~120k property examples. Judge arm 2 at M5 on that basis, or wait for M3c.

### 4.1 Held-out set — DECLARED 2026-08-28, before any run

Fills the slot `src/generalist/PLAN.md` §3.3 leaves open. Declared while no molecule result existed
to bias the choice — the only condition under which the declaration is worth anything.

**Held out from all molecule training, permanently:**

* **ClinTox** — one whole Tier-B dataset. Small (~1.5k, cheap to forfeit) and structurally unlike
  the rest of Tier B, being a toxicity/trial-failure endpoint rather than binding or permeability.
* **`bond_path`** — one whole Tier-A family. Chosen for the reason `direction` was chosen among the
  probes: it has a provable structural discriminator (SPD *is* the answer, by construction), so
  transfer there is unambiguous rather than a judgement call.

Both are in `registry.py`, enforced in code (`dataset.py` refuses them) rather than remembered.
Neither appears in any training mixture, in any arm, **including the specialist arm** — a
specialist run on a held-out task would make the arm-2-minus-arm-1 comparison meaningless for
exactly the task it matters most on.

Scored the two ways trunk plan §3.3 requires: zero-shot transfer, and adaptation efficiency
(steps-to-target from the trunk vs from base Llama). The second is expected to carry signal at 1B.

**Loss normalization (D7a).** Tier A answers are 1–3 tokens; Tier C captions are 50–100.
Per-example is the right default; if Tier C is included it is the second task in the repo (after
CLRS-Text) that may want the per-task escape hatch. Record the choice.

---

## 5. Two experiments that do not depend on beating anyone

Both cheap, both eval-only or nearly so, both producing claims that survive any leaderboard
outcome. Run them even if §2's gates fail.

**Size generalization.** Train on MoleculeNet-scale molecules (≤ ~35 heavy atoms), test on
peptides / macrocycles (~150 atoms). The CLRS protocol from trunk plan §7 applied to real
chemistry, and a far stronger claim than in-distribution accuracy. GTLM has a real shot: the flat
twin's SMILES grows linearly and its ring-closure bookkeeping degrades, while our bias is defined
identically at any N. Watch `max_spd` clamping — at 150 atoms paths exceed 32, and
`project-khop-spd-shortcut` records a k=8 break that was a `max_spd` artifact. §3.2.2's measurement
says the clamp is inert *at Tier-B sizes*; it says nothing at 150 atoms.

**Permutation invariance (§6).** Nearly free.

---

## 6. The free win: atom-order invariance

GTLM is permutation-equivariant over prefix nodes by Property 1, verified to 2.77e-5. **A
SMILES-based LLM is not.** The same molecule written from a different starting atom is a different
token string — which is exactly why SMILES augmentation exists as a standard trick.

The experiment: evaluate the flat twin on canonical SMILES and on 10 randomized SMILES per test
molecule; report the AUROC spread. GTLM's spread is provably zero. Cost: one extra eval pass.

This is a **property claim, not a leaderboard claim**, which is why it is worth more than three
AUROC points. It is also the cleanest molecular statement of the thesis: the graph arm answers a
question about a *molecule*, the flat arm about a *string that happens to denote one*.

**One constraint on the effect size, found at M0.** Randomisation only produces variation where the
molecule is topologically asymmetric — benzene has a single atom symmetry class, so every traversal
yields the same string and the flat arm is invariant *for free*. The measurement must be restricted
to, or stratified by, molecules with more than one symmetry class
(`Chem.CanonicalRankAtoms(mol, breakTies=False)`), or symmetric molecules dilute the flat arm's
spread toward zero and understate the effect. Pinned as a test.

---

## 7. Deferred, but do not let the schema preclude it: a 3D bias

The reason Uni-Mol and GEM win Tier B is 3D conformers. GTLM's bias is *any learned function of a
node pair* — so 3D is a drop-in entry in `BIAS_TYPES`:

```
b_3D(u, v) = MLP( RBF( ‖x_u − x_v‖ ) )      # radial basis expansion of interatomic distance
```

Same shape as `SPDBias`, no new machinery, SE(3)-invariant by construction since only the distance
enters. That would put GTLM on the same information footing as the 3D specialists **while keeping
the text channel**, which no 3D specialist has.

**Not in cycle 1.** Two things must be true now so it stays possible: the schema must carry optional
per-node coordinates, and the dataset builder must retain conformers rather than discarding them at
parse time. Both are free today and expensive to retrofit. Per `feedback-keep-core-gtlm-clean`, a
3D bias is a real `src/models/biases/` addition, not something absorbed in an experiment package.

---

## 8. Build order

| | Milestone | Done when | Cost |
|---|---|---|---|
| **M0** | ✅ **2026-08-28.** `rdkit` installed `--no-deps`; nine MoleculeNet CSVs from the DeepChem S3 bucket; scaffold split is ours (`scaffold_split`), avoiding a heavy `ogb` dependency that would have pulled on torch. Sizes, tokens, splits measured. | §3.1.1 | login node |
| **M1** | ✅ **2026-08-28.** `data.py`: RDKit `Mol` → networkx `DiGraph`, three encoding cells, flat SMILES serializer, scaffold split, `roundtrip_check`. 119 unit tests. | round-trip clean at each encoding's declared level, full corpus — §3.2.1 | CPU |
| **M2** | ✅ **2026-08-28 → 09-01.** Tier-A generator (`tasks.py`, 10 families), `dataset.py`, the experiment package, and eleven sweeps (~110 runs). Speed/memory settled (§3.1.2); recipe fixed (§3.2.7); bias ablated (§3.2.8); split defect found, fixed and fully re-measured (§3.2.10, `014`); closing table delivered (§3.2.5). | **5 graph wins / 2 ties / 2 flat losses / 1 void on molecule-disjoint Tier A**, split along *representational explicitness*; the bias is load-bearing and it is SPD | ~75 GPU-h |
| **M3-prep** | ✅ **2026-08-31.** The Tier-B scoring readout and example builder had **no tests** — `evaluate.py` is a port of relbench's and the tests were left behind in the port. Added `test_score_readout.py` (20) and `test_tier_b_examples.py` (25): known-AUROC synthetic logits, the sigmoid trap, tie-collapse instruments, label alignment against the BACE CSV, scaffold disjointness, and the ordering contract `load_data` slices on. Suite 216 → 221. | a silent scoring bug can no longer reach a headline | CPU |
| **M3-smoke** | ✅ **`010_tier_b_smoke`, 2026-08-31.** First GPU contact for the Tier-B path — **and it failed, exactly as intended** (§8.1). | the Tier-B path runs end to end | ~0.2 GPU-h |
| **M3a** | ✅ **2026-09-01. 18/18 COMPLETED** (`001_m3a_bace_bbbp.jsonc`). BACE + BBBP, both arms + `bias=none`, 3 seeds. Two cells were lost to a host OOM at `--mem 32G` and re-run at 64G; the config now carries 64G. | **Gate B1 FAILS on both sets** — the graph arm loses to its own flat twin (−0.017 / −0.033), exactly as §3.2.5 pre-registered. B2 and B3 both met, by both arms. §2.6 | ~12 GPU-h |
| **M3b** | ⬜ **HIV**, both arms, 3 seeds (`012` prep + `013` train, both written and verified, **neither submitted — held at the decision gate above**). Separated because it is 27× BACE and the tie-collapse stress case (~145 test positives). **The M3a diagnostic that gated it has been read and is clean**: `n_distinct` 67–87 of 152/204 and `tied_pair_fraction` 0.004–0.017, so the readout is not collapsing at this scale — but HIV is 27× larger and remains the stress case, so read the same two fields **before** its AUROC. Budget is set in *steps* (5 epochs ≈ 5140, matching Tier A's settled 5000) because 40 epochs here would be 41k steps and BACE's 1512 steps would be 1.5 epochs — neither is "the same budget". If the graph arm is `still_improving` at 5 epochs, re-run **both** arms at 10, and treat 5 as a lower bound. | the third anchor ladder filled in | ~25 GPU-h |
| — | **DECISION GATE. Provisionally NOT passed, and the call is mine to make, not this document's.** B1 needs ≥2 of 3 and has 0 of 2, so no HIV result can rescue it. There *is* no unexplained collapse against §2.2 — the opposite: B2 and B3 are both met (§2.6). So the gate's own rule says **the work is diagnosis, not more datasets**, and the diagnosis is already specified: M4's encoding sweep separates "the endpoints are motif-driven" from "Levi is the wrong graph for chemistry", which predict the same M3a result. **Held as of 2026-09-01** — M3b/`013` remains unsubmitted. | pass/fail recorded here | — |
| **M3c** | ⬜ Tox21 + SIDER (the deferred sets, §1), which also restores §4 arm 2's large multi-task set. Verify §2.3's anchors against the Uni-Mol PDF first. | multi-endpoint routing exercised | ~20 GPU-h |
| **M4** | ✅ **UNBLOCKED 2026-09-01.** The Tier-A re-run it was waiting on is done (`014`, §3.2.5) — the verdicts held and every graph margin grew. §3.2.5's *explicitness* hypothesis also gives the sweep a sharper job than "which of three cells wins": it predicts which families should separate them. Still outstanding from the re-run backlog: `007`'s bias ablation has never been measured on a clean split (§3.2.8). Then: encoding sweep, §3.2's **3 cells** (`rich×levi`, `terse×levi`, `rich×atom_only`) × `±smiles` × bias arms × 3 seeds, on the non-saturated Tier-A families + BACE/BBBP. Never on a saturated family, never at the stale recipe. Run `terse×levi` first — cheapest, and it decides whether the featurizer is needed at all. | §3.2's arms decided and written into a frozen config | ~100 GPU-h *(estimate)* |
| **M5** | ⬜ Multi-task arms 1 / 2 / 3 (§4). | arm 2 − arm 1 measured on Tier B | ~100 GPU-h *(estimate)* |
| **M6** | ⬜ §5 and §6 experiments. | size-generalization curve + permutation spread | ~20 GPU-h |
| **M7** | ⬜ Admission fork into the trunk, against §5's four criteria in the trunk plan. | pass/fail recorded in `lineage.json` | per trunk plan |

**Both outstanding cheap items ran** as `011_m2_loose_ends` (2026-08-31): a second seed on the four
`longest_chain` cells (§3.2.8's one stated hole) and gate A2's `stereo_assigned` ×
`stereo_tags: off` control, which had never been executed (§2.5). Running the latter is what
exposed §3.2.10.

**In flight as of 2026-09-01:** `015` (job 135484), the Tier-B budget check — the falsification test
for §8.2's fixed instrument, which says the extra budget is probably unnecessary — plus `019`
(job 135609), its three lost graph cells re-run. `014` (job 135331) is **done**, 20/20, and is
§3.2.5.

**`015` lost ALL FOUR of its GRAPH cells and none of its four flat cells.** 4/4 against 0/4 at the
same memory cap is the arm's footprint, not chance: the graph arm instantiates the structural-bias
tensors and the flat arm instantiates none, so a 64G cap the flat arm clears comfortably is one the
graph arm cannot.

| cell | failure | elapsed | reached |
|---|---|---|---|
| `bbbp`/graph/seed 0 | eval stall → cancelled | 3:02:32 | 3400/6120 |
| `bbbp`/graph/seed 1 | host-RAM OOM | 2:13:23 | 3200/6120 |
| `bace`/graph/seed 1 | host-RAM OOM | 2:47:45 | 3950/4440 |
| `bace`/graph/seed 0 | host-RAM OOM | 2:56:18 | **4300/4440 (97%)** |

Every failed cell reported MaxRSS 67048604K against the 67108864K limit. All four died mid-training
having produced no test number, so nothing was discarded that anyone could have preferred. `019`
re-runs three at 128G and `020` the fourth, which was still running when `019` went in.

**The result: `015` currently has ZERO graph-arm TEST data.** Its flat arm is complete and a clean
null, and that is worth little on its own — the graph arm is the one whose convergence was in
question. No conclusion about the budget can be written until `019`/`020` land.

#### What the failed runs' dev curves already say — and a confound they exposed

The four killed cells trained 52–97% of the way and left readable `trainer_state.json`, so their
validation trajectories survive. Best `eval_roc_auc` against M3a's, with the peak located as a
fraction of the 120-epoch budget (40 epochs = 33.3%):

| cell | best val | peak @ | M3a best val | Δ |
|---|---:|---:|---:|---:|
| `bace`/s0 | 0.7181 | 45.0% | 0.6915 | **+0.0266** |
| `bace`/s1 | 0.7098 | 30.4% | 0.6862 | **+0.0236** |
| `bbbp`/s0 | 0.9626 | 6.5% | 0.9647 | −0.0021 |
| `bbbp`/s1 | 0.9595 | 10.6% | 0.9628 | −0.0033 |

BBBP is settled: no gain, peaks in the first tenth. BACE looks like a real gain, and **three checks
say it is not evidence that the longer budget helped.**

1. **The flat arm is the control and it moved too.** Flat BACE dev rose +0.0155 / +0.0101 while its
   test went +0.0137 / −0.0309 — nowhere. A dev gain of this size demonstrably does not transfer on
   this dataset.
2. **Truncate to M3a's evaluation count and most of the gain is already there.** `best val` is a
   maximum over evaluations taken, and 120 epochs affords ~3× as many draws as 40; the max of 85
   noisy draws beats the max of 29 by construction. Cutting each curve to M3a's own count: `bace`/s1
   reaches **0.7098 — its entire full-run maximum — within the first 29 evaluations**, and `bace`/s0
   reaches 0.7079 of its eventual 0.7181. The gain is not bought by training longer.
3. **Because `num_epochs` also sets the LR schedule, these are not the same run truncated.**
   `lr_scheduler_type="cosine_with_min_lr"` spans the whole budget, so at matched steps the two runs
   are on different trajectories: **at step 1350, where `bace`/s1 peaked, the 120-epoch run is at
   2.449e-05 against the 40-epoch run's 3.529e-06 — 7× the learning rate.** The 40-epoch run has
   decayed to near its floor while the 120-epoch run is still near peak LR.

**So `015` does not cleanly isolate "budget" at all — it varies budget AND schedule shape together.**
On the dev evidence the extra *duration* buys nothing on either dataset; what BACE's graph arm
appears to like is the **gentler LR decay**, which is a different claim needing a different
experiment (same step count, schedule length varied). That experiment is not written. Until the test
numbers land this stays a dev-only reading of incomplete runs at one seed per cell, and it is
recorded here so the confound is not rediscovered later as a surprise.

**Resume was possible and was not used.** Three of the four killed cells left `checkpoint-N` with
full optimizer, scheduler, RNG and trainer state (`bbbp`/seed 1's was killed mid-write and has
none), and `bace`/seed 0's was at 97%. All four are being re-run from step 0 anyway, because
`resume_from_checkpoint` is not plumbed into `train.py` — zero references — and adding it while
other cells are using the harness is the wrong moment. The cost is real: roughly 3 GPU-hours on that
one cell, and it will recur at 120-epoch budgets against a tight ceiling. **Wire resume support
before the next long sweep.**

**A settings error in the first attempt at those re-runs is recorded here because it pointed the
wrong way.** `017`/`018` were written with `eval_steps 200`, copied from `014` (Tier A) instead of
from `015`, which uses **50** — as does `001`, the baseline every cell is compared against. That
governs how finely `load_best_model_at_end` can resolve the peak (~122 candidate checkpoints at 50,
~30 at 200), and coarser sampling can only miss the peak, never beat it. It would therefore have
depressed exactly the cells still outstanding, manufacturing evidence *for* §8.2's provisional
"the longer budget does not help". Both were cancelled before either wrote a record. Re-checking
`019` against the command line `015` actually emitted — not against its `.jsonc`, which cannot show
defaults — caught two more: `bias_lr` (`015` sets 5e-3, the default is 1e-3, so omitting it would
have run the bias channel at a fifth of the baseline's rate) and a stray explicit `stereo_tags`.
`019`'s emitted command is now byte-identical to `015`'s for the shared cell. **Diff the emitted
command, not the config, whenever a re-run has to match an existing sweep.**

**All four flat cells landed and the flat arm is a clean null:** BACE +0.014 / −0.031, BBBP −0.009 /
+0.017 — two up, two down, mean −0.002, every one peaking inside the original 40-epoch budget and
the BBBP pair at 2–3% of the tripled one. That is also the control for the one live counter-argument
in this sweep: the BACE **graph** cells reached validation ~0.025 above M3a, which has the shape of
"40 epochs was tight". The flat arm's validation rose too (+0.010, +0.016) and its test scores went
nowhere, so a validation gain of that size is **not predictive of test here**, and the graph arm's
test numbers are what will settle it.

**`015` ran the whole array at 99.91% of its memory cap**, and that is a standing hazard for every
config in this directory, not an incident in one. Measured 2026-09-01 against a 64 GiB cgroup limit
(67108864K): the two completed runs peaked at 67047112K, the killed run at 67048984K, and all five
then-running runs at 67048604K — **on two different nodes, within 2 MB of each other**. A figure that
identical across nodes is not process growth; it is the cgroup filling with reclaimable page cache
from the dataset artifacts, which is harmless right up until an allocation spike outruns reclaim.
Two cells won that race and one lost. So `bbbp`/graph/seed 1 was not a cell with a memory problem —
it was the cell whose spike landed first, and reading the failure as a property of that cell would
be reading noise as signal. `014` and `015` had already been raised 32G → 64G after M3a's OOMs; the
lesson is that the *margin*, not the limit, is what has to be checked, and 64G leaves ~59 MB of it.
ana had ~507 GB free at the time. **New Tier-B configs should ask for 128G**, and a run's `MaxRSS`
against `ReqMem` is worth reading before concluding a sweep was clean.

**`015` also lost a cell to a STALL, and that is the failure worth learning from.** Task 0
(`bbbp`/graph/seed 0) ran normally for 1h53m and 3450 of 6120 steps, completing 102 periodic
evaluations at ~5 s each. Evaluation 103 then took 53:54 to reach batch 15 of 51, degrading
monotonically — 226 → 255 → 281 → 305 → **345 s/batch**, roughly 4000× its own established rate.
**The job stayed `RUNNING` and its log kept being written the whole time**, so job state, log
freshness, and `MaxRSS` all reported health; only comparing the step counter against its own history
showed anything wrong. It was cancelled once the arithmetic was decisive rather than left to hit the
wall: 1.98 h of limit remained, the current evaluation alone needed 2.26 h at the best rate seen
during the stall and 3.45 h at the latest, and 43% of training was still ahead. Host RSS was
byte-identical to the two healthy graph runs beside it, so the page-cache story above does **not**
explain this one; `ixh` was shared with three jobs from another user, one 5 days old, which makes
contention the leading explanation — **leading, not established**, since it was diagnosed from
outside the node. A stall watcher now compares each running job's step counter against its own
previous value and reports no movement over ~20 minutes, which is the check that would have caught
this in a fifth of the time it actually took.

**Written and NOT submitted, all three held pending a decision:** `012`/`013` (M3b/HIV, behind the decision gate
above), and `016_leakage_detector_restore` (§9) — the last of which is the one with a standing
argument for jumping the queue, since until it runs the suite has no leakage detector and every
result in this document rests on splits nothing is currently checking.

### 8.1 What the Tier-B smoke caught (2026-08-31)

Both smoke runs died in 55 s with `KeyError: 'answers'` at `dataset.py:245`. The Tier-A generator
puts an `answers` counter in its stats dict; `build_tier_b_examples` never did, and the key was
read **unguarded in a progress `print`** — so a log line failed the job, *after* the `.gtds` and its
meta sidecar had already been written to disk. Had 001 gone first, this would have taken out all 18
runs, and the half-written artifacts would have been silently reused by the re-run.

**The quieter half is the one worth remembering.** `train.py::_answer_stats` reads the same key
with `.get`, so had the print not crashed, every Tier-B run would have recorded **`base_rate:
null`** — the field §3.2.4.1 made mandatory precisely because a score without its floor is
uninterpretable. A loud crash was hiding a silent integrity gap, and only the loud one would have
been noticed.

Fixed three ways, all pinned by regression tests: Tier-B stats now carry `answers` **and**
`answers_by_split`; `_answer_stats` takes the floor from the **test** split when the breakdown
exists, recording `base_rate_source` so a reader is not left guessing (Tier A is unchanged — its
splits are drawn from one generator, so the two coincide); and the print is `.get`, because a log
line must never be what fails a job.

This is the second time in this campaign that a cheap first-contact run paid for itself
(`000_smoke` settled flex-vs-eager). **Keep smoking a new path before committing a sweep to it.**

### 8.2 `still_improving` was inverted — every "budget-limited" reading in this document is void (found 2026-09-01)

**The instrument reported overfitting as under-training.** `_convergence` reads `tail_gain =
values[-1] - values[-4]` off the eval curve. But `_eval_curve` was called at the *end* of the run,
and by then two things had happened: `trainer.train()` had restored the best checkpoint
(`load_best_model_at_end=True`), and `trainer.evaluate(..., metric_key_prefix="eval")` on the line
after it had re-scored **that** model on val and logged it into the same history. The curve
therefore ended on a point equal, by construction, to its own maximum.

So `tail_gain = max(curve) - values[-4] ≥ 0` **always**, and `still_improving` fired hardest on the
runs that had fallen *furthest* from their peak. It is not a weak signal; it is close to an
inverted one.

**Measured on M3a's 16 records.** The final curve point equals the maximum in **16 of 16**. The
flag says 14 of 16 runs were still improving. Recomputed from the same stored curves with that
point dropped: **2 of 16**.

| set | where val actually peaked (fraction of budget) | reading |
|---|---|---|
| BACE | 0.39 – 0.75 of 29 evals | converged inside the budget, then declined |
| BBBP | 0.03 – 0.54 of 40 evals, six of seven ≤ 0.31 | peaks in the **first fifth**; the rest of the run is overfitting |

A worked case — `bace/flat/seed 0`: val ROC-AUC peaks at **0.7432 at step 750** and falls to 0.6695
by step 1450 while `eval_loss` climbs 0.77 → 2.20. Textbook overfitting. The flag called it
`still_improving: True`.

**What this voids.** Every "this score is a lower bound, not a ceiling" note sourced from
`still_improving` — including the framing that M3a needs re-running at a longer budget. It does
not: no BACE or BBBP cell was budget-limited, and more epochs cannot raise a number that is already
selected from the peak. `load_best_model_at_end` means the *reported* scores were never harmed;
only the convergence claim about them was.

**What it does not void.** §3.2.7's finding that `longest_chain` gained +0.104 after the flag said
converged. That is the opposite failure — the flag reading `False` when a run was still climbing —
and a curve ending on its own maximum makes `tail_gain` too *large*, never too small. So a `False`
was, if anything, understated, and §3.2.7's direct measurement stands on its own.

**Fixed** by snapshotting the curve immediately after `trainer.train()` and before either
`evaluate` call — exact, rather than filtering a duplicate point heuristically. `_convergence` also
now records **`peak_fraction`** (best eval index ÷ curve length), which answers "was there budget
left?" directly and needs no noise threshold.

`tests/experiments/molecules/test_convergence.py` (11 tests) pins the arithmetic and, in
`test_the_flag_inverts_on_a_contaminated_curve`, reproduces the defect: the same declining
trajectory flips from `still_improving: False` to `True` when the post-reload point is appended.
**`_convergence` had no tests at all** — the third instrument in this campaign to be wrong in a
direction nobody checked, after §3.2.10's split and §8.1's `base_rate`. The pattern is not bad
luck: a diagnostic that is only ever *read*, never *asserted on*, has no error-detecting surface.

**Blast radius checked: molecules only.** Nine experiments set `load_best_model_at_end`, but
`still_improving` / `tail_gain` / `eval_curve` appear nowhere outside `molecules/train.py`
(`grep` over `src/`), so no other experiment derives a convergence claim from `log_history` and none
inherits this defect. The pattern to watch for elsewhere is the general one — *reading
`trainer.state.log_history` after a post-training `evaluate` call* — not these field names.

> Records written before 2026-09-01 carry the old flag. Do not read `still_improving` out of them;
> recompute it from `eval_curve`, which is stored in full in every record and is what the tables
> above were rebuilt from. **`014` is split across the fix**: `train.py` was corrected at 01:14:35
> and array tasks **0 and 1** had already started (01:12:04), so those two records carry the old
> inverted flag and no `peak_fraction`; tasks 2–19 carry the corrected one. Presence of
> `peak_fraction` is the exact discriminator — test on that, not on a timestamp, and never on
> whether the curve happens to end on its maximum (a genuine run may). `dataset.py`'s
> `repeats_by_split` was added *before* the array started, so every `014` record has it.

### 8.3 Tier-B val and test are different populations — not two draws from one (measured 2026-09-01)

Found while checking whether M3a's budget was short. It is not a bug, and the split is **not** going
to be changed without a decision, because it is DeepChem's — but it changes how every Tier-B number
here is read.

**The observation.** M3a's BBBP runs score val ROC-AUC ≈ **0.965** and test ROC-AUC ≈ **0.70**. Both
splits are 52–55% positive, so a base-rate artifact is ruled out.

**The mechanism, measured.** `scaffold_split` groups by Bemis–Murcko scaffold and pours groups into
train → val → test, sorted by `(len(group), group[0])` **descending** — DeepChem's exact ordering.
Two consequences that are invisible until you look:

1. Train takes every multi-molecule scaffold group; **val and test are 100% singleton scaffolds**
   (BACE 151/151 and 152/152, BBBP 204/204 and 204/204). So val is not "held-out training
   chemistry" and test is not "harder than val" by group size — that hypothesis was checked and is
   false.
2. Among singletons the tiebreak is `group[0]` descending, i.e. **original CSV row order**. Val and
   test therefore come from *disjoint contiguous slices of the file*:

| set | split | original row range | positive rate | mean SMILES length |
|---|---|---:|---:|---:|
| BBBP | train | 0 – 2038 | 0.822 | 44.6 |
| BBBP | **val** | **716 – 1196** | 0.549 | 68.5 |
| BBBP | **test** | **5 – 714** | 0.525 | 53.1 |
| BACE | train | 1 – 1512 | 0.426 | 63.0 |
| BACE | **val** | **396 – 1011** | 0.556 | 64.9 |
| BACE | **test** | **0 – 381** | 0.605 | 73.0 |

MoleculeNet's CSVs are ordered by source series, so these are different chemical populations, and
they differ measurably on a trivial proxy (SMILES length shifts by 15 characters on BBBP).

**What follows, and what does not.**

- **Comparability with §2.2 is preserved, and that is the main thing.** Every published scaffold-split
  number we anchor against was produced by this same DeepChem ordering. Replacing it with something
  "fairer" would silently break the comparison the anchor table exists to make.
- **Checkpoint selection is the casualty.** `metric_for_best_model=eval_roc_auc` selects on val, which
  is a different population from the one reported, and on BBBP that signal saturates by **epoch 2**
  and then varies by an sd of 0.002–0.009. The selected checkpoint is close to arbitrary among the
  post-saturation ones. This is a **selection** problem; it is not fixed by a longer budget and 015
  does not address it.
- **It bounds how a val/test gap should be read.** A model that scored 0.96 on val and 0.70 on test
  has not necessarily overfit by 26 points; some of that gap is the population change and would
  appear for a perfectly regularised model.

**Not acting on this tonight — it is a design decision, not a defect.** The options, for the record:
keep DeepChem's split and accept noisy selection (status quo, maximum comparability); select on a
held-out slice of *train* instead of val; or report both the val-selected and last-checkpoint test
scores so the selection's contribution is visible. The third is cheap and additive and is the one
worth doing first.

**M1's round-trip test is the highest-value test in the plan.** An encoding bug — a dropped aromatic
flag, a bond mapped to the wrong Levi node — is otherwise completely silent: the model trains, the
loss falls, the number is just mediocre, and six weeks later it looks like an architectural
limitation. The molecular equivalent of relbench's `test_evaluate_crosscheck.py`.

**Total ≈ 275 GPU-h** *(estimate)* — small next to Phase 1's 300–500. Per `feedback-submit-to-slurm`,
nothing runs on the login node except M0's download and stats.

---

## 9. Risks

* **Framing.** Reported as "GTLM on MoleculeNet", the honest outcome (lose to Uni-Mol, beat our flat
  twin) reads as failure. §0 and §2.5 exist to fix the frame in advance.
* **RESOLVED — Tier A's test split was not molecule-disjoint** (§3.2.10). Fixed 2026-08-31
  (§3.2.10.1) and fully re-measured 2026-09-01 (`014`, §3.2.5): the verdicts held and **every graph
  margin grew**. Kept in this list because the *lesson* outlived the defect. Tier-A generation had
  **no test coverage at all**, every arm was affected equally, so nothing looked anomalous until a
  control that *predicted* chance came back at 0.936. **Untested data plumbing is where this class of
  defect lives**, and it has now produced three separate defects in one campaign — the split
  (§3.2.10), `base_rate` (§8.1) and the convergence flag (§8.2). The common shape: a quantity that is
  only ever *read*, never *asserted on*, has no error-detecting surface. The standing rule that came
  out of it — every instrument gets a test that fails when the instrument is wrong, including one
  that reproduces the defect.
* **A pool wide enough to fix the split can destroy a family.** `014` widened the molecule pool
  because the fix made the old one infeasible, and that pushed `stereo_assigned`'s test split to a
  single answer — both arms score 1.000 and the family is void (§3.2.10.1). Two families lost most
  of their headroom as well. **Changing what a benchmark samples is a benchmark change**; check base
  rates per family after any pool edit, and read `degenerate_test_split` in the run record.
* **Tier A does not predict Tier B, and the one signal we have is negative.** §3.2.5's split puts
  binding/permeability/toxicity on the side SMILES represents well, where the graph arm loses — and
  **M3a confirmed it** (§2.6, gate B1 fails on both BACE and BBBP). Tier A labels are
  deterministic functions of the graph we hand the model; Tier B labels are noisy experimental
  measurements of properties partly not determined by the 2D graph at all. Do not read the M2
  closing table as evidence about MoleculeNet.
* **The negative control is now WORSE than never firing.** `stereo_assigned` used to saturate in
  both arms (0.997 / 0.996); on `014`'s pool its test split has a single answer, so it scores exactly
  1.000 either way and is void (§3.2.10.1). The suite has **no working leakage detector at all**, and
  the `stereo_tags: off` contrast that would restore it cannot be run on this pool. Until then, the
  thing that caught §3.2.10 is not in the suite.

  **The restoring pool is now measured, not just nominated** (2026-09-01, CPU only — the generator
  run over each scaffold split of each candidate pool):

  | pool | test split: examples / classes / base | |
  |---|---|---|
  | `bace,bbbp,tox21,lipo` | 1000 / 10 / **0.732** | usable — 0.268 of headroom |
  | `hiv,bace,bbbp,tox21,lipo` | 1000 / **1** / 1.000 | degenerate, as `014` hit |

  Two things that settled beyond the headline. First, 0.732 is within 0.017 of the 0.715 base `011`
  ran against, so the control's **dynamic range is preserved** and `011`'s 0.9928 → 0.8152 remains a
  usable reference point; a formally-fine base of 0.95 would have left no room to fall. Second, on
  the five-corpus pool the **train** split is 0.978 base (3911 of 4000 one answer) — `014`'s
  `stereo_assigned` cell had almost nothing to learn from either, so it is void twice over.

  `016_leakage_detector_restore` is written against that pool and **pre-registers the reading**:
  `off` at or below **0.774** (base + 3σ, σ = 0.014 at n=1000) restores the detector and passes gate
  A2's second half; materially above it means information reaches the model by an unintended route
  *even on molecule-disjoint splits*, i.e. §3.2.10.1 closed one leak and not the class of leak —
  which would block M4 and outrank anything in the M2 table. **Not submitted; held (§8).**
* **AUROC tie collapse on HIV.** 3.5% positives plus bf16 margin quantization to 1/8
  (`project-gtlm-margin-quantization`) is the worst case for the margin readout. `n_distinct` and
  `tied_pair_fraction` in every record from M3 onward, not retrofitted.
* **Scaffold-split variance.** Small sets swing by several AUROC points across seeds. ≥3 seeds,
  mean ± sd, no per-dataset cherry-picking — the survey's instability finding (§2.4) is the
  evidence for this.
* **Anchor incomparability.** InstructMol is 7B with a pretrained graph encoder; Uni-Mol is 3D +
  10M-molecule pretraining; Mol-LLM is a multi-task generalist. Almost all are fine-tuned per
  dataset (§2.1). **Our matched control is our own flat twin.**
* **Our own arms are not equally tuned** (§3.2.5 disclosure 2). Until a flat `lr` sweep is run, the
  claim is "beats our flat twin", never "beats a fully-tuned baseline".
* **Levi changing the spectrum** — mitigated by the `atom_only` arm being in M4, not a follow-up.
* **Do not call a floor early** (`feedback-dont-call-floors-early`). Realised **twice** in this
  campaign: M2's 0.000 was undertraining, and M2c's "plateau" at 0.855 gained 10 points at the same
  settings. The `still_improving` flag installed after the first occurrence did not catch the
  second.
* **An inherited recipe is a confound until it is checked.** §3.2.7 spent ~30 GPU-h and four sweeps
  diagnosing an architectural deficit that was a copied hyperparameter block from the oldest
  experiment in the repo. The check costs one `grep` across `runs.jsonl` files and belongs **before**
  the first diagnostic sweep in any new domain, not after the fourth.
