# Molecules × GTLM — implementation plan

**Status:** M0–M2 complete (2026-08-30). **M3 is next, and it is scoped to BACE / BBBP / HIV**
(§1 Tier B, DV 2026-08-31). Written to run **before** the generalist trunk
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
| `001_bace` | 9 | M3 Tier B — **not yet run**; needs the tuned recipe (§3.2.7) | 8 |
| `002_difficulty` | 18 | Tier-A difficulty screen at 1250 steps | 3.2.4 |
| `003_hardware_check` | 4 | H100/A100 run the flex compile path | — |
| `004_extended` | 20 | the screen at 2500 steps, matched pairs | 3.2.4 |
| `005_fgcount_ablation` | 3 | bias/question-node ablation — **result retracted** | 3.2.6 |
| `006_recipe` | 8 | **the recipe fix; `longest_chain` flips** | 3.2.7 |
| `007_tuned_ablation` | 4 | **the win is the bias, and the bias is SPD** | 3.2.8 |
| `008_m2_screen_graph` | 7 | M2 closing screen, graph arm at `bias_lr 1e-2` | 3.2.5 |
| `009_m2_screen_flat` | 5 | M2 closing screen, flat arm (its control twin) | 3.2.5 |

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

### Tier B — MoleculeNet ⚠️ SCOPED 2026-08-31 (DV)

Scaffold split (ours, `scaffold_split`), ROC-AUC. Built at M3 (`tier_b.py`); GPU runs pending.
Molecule counts are post-drop; *examples* are `(molecule, endpoint)` pairs after skipping absent
labels.

**Primary set — run these three, and nothing else, until they have reported:**

| Dataset | molecules | endpoints | examples | positives | scaffold split |
|---|---:|---:|---:|---:|---|
| **BACE** | 1513 | 1 | 1513 | 45.7% | 1210/151/152 |
| **BBBP** | 2039 | 1 | 2039 | 76.5% | 1631/204/204 |
| **HIV** | 41119 | 1 | 41119 | **3.5%** | 32895/4112/4112 |

**The reason for the scope is the anchors, not the cost** (DV, 2026-08-31). These three are the
only Tier-B sets with a *complete anchor ladder* — 3D specialist, graph-LLM at 7B, tuned-LLM at
7B, and a prompted-LLM floor (§2). That ladder is what converts one AUROC into a position in the
space of solutions. Tox21 and SIDER have a single anchor row between them, so a number on them is
uninterpretable in exactly the way §0 warns about. They are also all single-endpoint, so the first
Tier-B measurement does not simultaneously test the multi-endpoint routing machinery.

**Order them BACE → BBBP → HIV.** HIV is 27× BACE's size and the imbalance stress case: ~145
positives in its test split, so a handful of distinct margins decides the AUROC outright.
`n_distinct` and `tied_pair_fraction` are in the run record from the first run for exactly this.
BACE and BBBP settle whether the pipeline works; HIV is a stress test that follows.

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
| Mol-LLM | The genuine generalist: one model over a broad task suite (SELFIES + hybrid GINE/TokenGT encoder + Q-Former), trained with next-token prediction plus a structure-preference objective |

### 2.2 The primary three — MoleculeNet classification, scaffold split, ROC-AUC ↑

InstructMol (arXiv:2311.16208, **current version**, Table 2; 3 random seeds, scaffold splits).
Primary source, read 2026-08-31.

| | BACE | BBBP | HIV | what it is |
|---|---:|---:|---:|---|
| DMP (TF+GNN) | 89.4 | 77.8 | 81.4 | strongest row in the table; not a like-for-like |
| **Uni-Mol** | **85.7** | **72.9** | **80.8** | **the ceiling, not the target** — 3D conformers, 10M-molecule pretraining |
| MolFM | 83.9 | 72.9 | 78.8 | |
| GraphMVP-C | 81.2 | 72.4 | 77.0 | |
| MolCA (1D+2D) | 79.8 | 70.0 | — | |
| KV-PLM | 78.5 | 70.5 | 71.8 | |
| MoMu | 76.7 | 70.5 | 75.9 | |
| GraphCL | 75.3 | 69.7 | 78.5 | |
| ChemBERTa-2 | 73.5 | 69.8 | 79.3 | |
| **InstructMol-G** (7B + graph tokens) | **84.3 ±0.6** | **68.6 ±0.3** | **74.0 ±0.1** | **the graph-LLM band we should land in** |
| InstructMol-GS (7B + graph + SMILES) | 82.1 ±0.1 | 72.4 ±0.3 | 68.9 ±0.3 | note it is *not* uniformly better than -G |
| **Llama-2-7B-chat, LoRA** | **74.8** | **65.6** | **62.3** | **the row to beat, at 1/7 the parameters** |
| Vicuna-v1.3-7B, LoRA | 68.3 | 60.1 | 58.1 | |
| Galactica-6.7B | 58.4 | 53.5 | 72.2 | |
| Galactica-30B | 72.7 | 59.6 | 75.9 | |
| Galactica-120B | 61.7 | 66.1 | 74.5 | |
| Vicuna-v1.5-13b-16k, 4-shot | 49.2 | 52.7 | 50.5 | the prompted-LLM floor — near chance |

**Version drift, checked and resolved.** ar5iv still serves an earlier version of InstructMol with
InstructMol-G at 85.9 / 64.0 / 74.0 — materially different on BBBP. The numbers above are the
current arXiv version. Do not "fix" this table against the stale mirror.

### 2.3 Anchors for the deferred sets

Needed only if §1's deferred sets re-enter at M3c. Uni-Mol, ROC-AUC, scaffold split:

| | Tox21 | SIDER | ClinTox |
|---|---:|---:|---:|
| Uni-Mol | 79.6 | 65.9 | 91.9 |

**Provenance caveat — these three are secondary.** The Uni-Mol paper's own table could not be read
directly (OpenReview serves a bot-check page to the fetch path); they come from a table that
quotes it. Its BACE/BBBP/HIV column reproduces InstructMol's Uni-Mol row exactly (85.7 / 72.9 /
80.8), which is a consistency check but not a substitute. **Verify against the Uni-Mol PDF before
these appear in a writeup.** No LLM-baseline anchors exist for these sets at all — part of why
they are deferred. The regression sets have no anchors of any kind yet.

### 2.4 Two anchors not to calibrate against

* **Mol-LLM** (arXiv:2502.02810) reports BACE 80.5 and **BBBP 81.1**. The BBBP figure is far out of
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
| **A1** | Tier A: graph arm ≥ +15 points over the SMILES flat twin on `ring_membership`, `ring_size`, `bond_path`. | **Not met.** Best is `ring_size` +11.5. The graph arm sits at 0.99–1.00 on these families, so the gap is bounded by flat's headroom, not by the graph channel. §3.2.5. |
| **A2** | Tier A: graph arm **wins** `stereo_potential`; on `stereo_assigned`, chance under `stereo_tags: off`, high under `on`. | **Half met.** `stereo_potential` +0.025 (2.6σ) ✅. The `stereo_assigned` off/on contrast was never run and both arms saturate at 0.997 — the negative control did not fire. §3.2.5. |
| **B1** | Tier B: graph arm beats **our own flat twin** by ≥ 1 sd on ≥ 2 of the primary 3 classification sets. **This is the real result.** | Pending M3. *(Threshold restated from "4 of 6" for the scoped Tier B.)* |
| **B2** | Tier B: within 3 AUROC of InstructMol-G on BACE/BBBP/HIV at 1B vs 7B. Aspirational, not a gate. | Pending M3 |
| **B3** | Tier B: beat the Llama-2-7B-chat LoRA row (74.8 / 65.6 / 62.3) at 1B. If we do not, the flat twin will not either, and the domain is telling us the molecules are being read badly — check M1's round-trip test before blaming the architecture. | Pending M3 |
| **P1** | §6: flat twin AUROC spread across randomized SMILES ≥ 2 points; GTLM spread < 0.1 (bounded by Property 1's 2.77e-5). | Pending M6 |
| **Null** | If the graph arm does not beat flat on **Tier A**, the *encoding* is wrong, not the architecture. Do not theorize about the model until the round-trip test and a `bias=none` control have both been run. | Both run. Round-trip clean (§3.2.1); `bias=none` measured (§3.2.8). |

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

**Pre-registered prediction (DV, 2026-08-29, before M4 runs): `rich × levi` wins, and its margin
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

**DECISION 2026-08-29 (DV): `node_position_mode` is unwired from this experiment entirely.** No
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

**Renamed 2026-08-29 (DV): the values are `"on"` / `"off"`, not `"isolated"` / `"off"`.** graphqa
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

### 3.2.5 M2 CLOSING TABLE — the Tier-A result (`008` + `009`, jobs 134481/134482, 2026-08-30)

Twelve runs, all COMPLETED: seven graph families at `lr 3e-5, bias_lr 1e-2, lora_r 16, 5000 steps`,
five flat twins at the same recipe, plus `fg_count` and `longest_chain` flat reused verbatim from
`006_recipe` (exact — the flat arm instantiates no bias parameters, so `bias_lr` cannot reach it).

| family | base | **graph** | **flat** | Δ | σ | verdict |
|---|---:|---:|---:|---:|---:|---|
| `ring_size` | 0.498 | **0.992** | 0.877 | **+0.115** | +10.7 | **graph** |
| `longest_chain` | 0.253 | **0.991** | 0.947 | **+0.044** | +5.7 | **graph** |
| `stereo_potential` | 0.285 | **0.962** | 0.937 | **+0.025** | +2.6 | **graph** |
| `ring_membership` | 0.505 | **1.000** | 0.980 | **+0.020** | +4.5 | **graph** |
| `fg_atom_membership` | 0.503 | 0.986 | 0.974 | +0.012 | +1.9 | tie |
| `fg_presence` | 0.760 | 0.918 | 0.962 | −0.044 | −4.2 | flat |
| `fg_count` | 0.760 | 0.888 | 0.936 | −0.048 | −3.8 | flat |

**4 graph wins, 1 tie, 2 flat wins.** Mean Δ +0.018, median +0.020. Three further families are
reported from `004_extended` and were not re-run because both arms sit at the ceiling and no recipe
change can move one: `aromatic_ring` (1.000 / 0.997), `ring_count` (0.993 / 0.993),
`stereo_assigned` (0.997 / 0.996 — the designed-in loss, which therefore never fired).

**The line is topology versus chemical-motif recognition.** The graph arm wins **every topological
question** — ring size, ring membership, path length, stereocentre potential — and loses **both
functional-group questions**. It is *not* local-vs-global: `longest_chain` and `ring_size` are
whole-molecule properties and the graph arm wins them decisively. Nor is it distance-vs-aggregation
(§3.2.8's hypothesis): `stereo_potential` is a subtree-comparison question with no distance reading
and the graph arm wins it. Recognising a nitro group is pattern-matching over atom types and local
bonds, which a SMILES string represents natively and compactly and which the Levi transform
scatters across nodes and edges. `fg_atom_membership` sits exactly on the boundary and lands at a
tie: it is the motif question re-asked about a *named atom*, and naming the atom recovers most of
the gap.

**§3.2.8's distance hypothesis is falsified as stated, and the failure was pre-registered.** The
prediction table was written into this document before the runs. It called `ring_size`,
`ring_membership` and `longest_chain` correctly as wins and `fg_count`/`fg_presence` correctly as
losses, but predicted **no benefit** on `stereo_potential` — designated the sharpest test precisely
because gate A2 predicted the opposite. Gate A2 was right. Do not rescue the distance framing by
reclassifying `stereo_potential` after the fact; replace it with *topology*, which covers all four
wins without special pleading.

**`bias_lr` 5e-3 → 1e-2 is a verified null.** `longest_chain` 0.989 → 0.991 (0.5σ), `fg_count`
0.897 → 0.888 (0.7σ), with `bias_norm` rising ~2.4× (40 → 104, 40 → 96) — so the module trained to
a very different magnitude and accuracy did not move (`feedback-verify-nulls-are-real` satisfied).
**Keep 5e-3 as the settled recipe**; treat 1e-2 as measured-equivalent, not preferred. The
practical consequence is that the five families measured at 1e-2 are directly comparable to every
5e-3 number in §3.2.7–§3.2.8.

**Three disclosures that belong beside this table wherever it is quoted.**

1. **One seed.** Every σ is a two-proportion sampling bound at n = 1000 and says nothing about
   seed-to-seed training variance. The marginal cells (`fg_atom_membership` +1.9σ,
   `stereo_potential` +2.6σ) are the ones a second seed could move.
2. **The arms are not equally tuned.** `bias_lr` is graph-arm-only by construction, so this is a
   tuned arm against an untuned one. The flat arm keeps `lr 3e-5` inherited from graphqa, and 006
   measured it responding to `lr` (+0.029 on `fg_count`). A flat `lr` sweep is what would earn the
   claim "beats a fully-tuned baseline"; **until then, do not make that claim.**
3. **The baseline got much stronger and the win survived it.** The tuned recipe moved the flat arm
   up 3–6 points on every re-run family (`ring_membership` 0.941 → 0.980, `fg_presence` 0.902 →
   0.962, `fg_atom_membership` 0.938 → 0.974, `stereo_potential` 0.907 → 0.937, `ring_size` 0.827 →
   0.877). The graph wins are measured against that, not a weaker control.

**What this table predicts about Tier B — write it down before M3 runs.** The split says the graph
arm wins topology and loses motif recognition. BACE binding, BBBP permeability and HIV activity are
**pharmacophore/motif-driven**, so our own Tier A result puts the graph arm on the *losing* side of
its own split at Tier B. Recording that now is what makes the M3 result interpretable in either
direction; discovering it afterwards would be a rescue.

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

**Open, and cheap.** Every cell here is one seed. 5.2σ bounds *sampling* error only. A second seed
on the four `longest_chain` cells is the last thing standing between this and a claim fit for
`src/generalist/PLAN.md`.

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
| **M2** | ✅ **2026-08-28 → 08-30.** Tier-A generator (`tasks.py`, 10 families), `dataset.py`, the experiment package, and ten sweeps (~90 runs). Speed/memory settled (§3.1.2); recipe fixed (§3.2.7); bias ablated (§3.2.8); closing table delivered (§3.2.5). | **4 graph wins / 1 tie / 2 flat losses on Tier A**, split along topology vs chemical motif; the bias is load-bearing and it is SPD | ~50 GPU-h |
| **M3a** | ⬜ **NEXT. BACE + BBBP**, both arms, 3 seeds, tuned recipe, `graph_only`. `tier_b.py` + `evaluate.py` are built (scaffold split, one example per `(molecule, endpoint)`, endpoint in the QUESTION node; the relbench margin readout ported — `logit(" Yes") − logit(" No")` in fp32, sigmoid before threshold metrics, `n_distinct` / `tied_pair_fraction` in every record). `001_bace.jsonc` needs its recipe updated to §3.2.7's before it runs. | gate **B1** decided on two sets; B2/B3 read against §2.2 | ~10 GPU-h |
| **M3b** | ⬜ **HIV**, both arms, 3 seeds. Separated because it is 27× BACE and the tie-collapse stress case (~145 test positives). Read `n_distinct` and `tied_pair_fraction` **before** reading the AUROC. | the third anchor ladder filled in | ~25 GPU-h |
| — | **DECISION GATE (DV).** Proceed only if the primary three are satisfying: B1 met on ≥2 of 3, and no unexplained collapse against §2.2's LLM rows. If they are not, the work is diagnosis, not more datasets. | pass/fail recorded here | — |
| **M3c** | ⬜ Tox21 + SIDER (the deferred sets, §1), which also restores §4 arm 2's large multi-task set. Verify §2.3's anchors against the Uni-Mol PDF first. | multi-endpoint routing exercised | ~20 GPU-h |
| **M4** | ⬜ Encoding sweep: §3.2's **3 cells** (`rich×levi`, `terse×levi`, `rich×atom_only`) × `±smiles` × bias arms × 3 seeds, on the non-saturated Tier-A families + BACE/BBBP. Never on a saturated family, never at the stale recipe. Run `terse×levi` first — cheapest, and it decides whether the featurizer is needed at all. | §3.2's arms decided and written into a frozen config | ~100 GPU-h *(estimate)* |
| **M5** | ⬜ Multi-task arms 1 / 2 / 3 (§4). | arm 2 − arm 1 measured on Tier B | ~100 GPU-h *(estimate)* |
| **M6** | ⬜ §5 and §6 experiments. | size-generalization curve + permutation spread | ~20 GPU-h |
| **M7** | ⬜ Admission fork into the trunk, against §5's four criteria in the trunk plan. | pass/fail recorded in `lineage.json` | per trunk plan |

**Cheap and outstanding, worth folding into M3a's submission:** a second seed on the four
`longest_chain` cells (§3.2.8) and the `stereo_assigned` × `stereo_tags: off` run that gate A2
needs and that has never been executed (§2.5). Both are one job each and both close a stated hole.

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
* **Tier A does not predict Tier B, and the one signal we have is negative.** §3.2.5's split puts
  binding/permeability/toxicity on the motif side, where the graph arm loses. Tier A labels are
  deterministic functions of the graph we hand the model; Tier B labels are noisy experimental
  measurements of properties partly not determined by the 2D graph at all. Do not read the M2
  closing table as evidence about MoleculeNet.
* **The negative control never fired.** `stereo_assigned` saturates in both arms (0.997 / 0.996), so
  the suite currently has no working leakage detector. The `stereo_tags: off` run is what restores
  it (§8).
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
