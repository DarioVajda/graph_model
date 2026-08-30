# Molecules × GTLM — implementation plan

**Status:** M0–M2c done, 2026-08-29. Written to run **before** the generalist trunk
(`src/generalist/PLAN.md` §7 names molecules as an admission-gate domain and leaves the
representation decision open — this document is that decision).

**Read §3.2.7 first.** Every result in §3.2.4–§3.2.6 was measured under a training recipe
inherited from `expressiveness` that turned out to be the lowest-capacity one in the repo.
Retuning it flipped `longest_chain` from −0.079 to **+0.042** in the graph arm's favour and
invalidated §3.2.6's "the bias is inert" null. Those sections are kept because the reasoning in
them is still how the campaign should be run; their **numbers are lower bounds for the graph
arm**, not measurements of it.

**Configs.** Every sweep in this document is reproducible from `configs/`, one file per sweep,
renumbered contiguously 000–006 on 2026-08-30:

| config | runs | what it settled | §
|---|---:|---|---|
| `000_smoke` | 6 | plumbing, flex-vs-eager, s/it and peak GB | 3.1.2 |
| `001_bace` | 9 | M3 Tier B — **not yet run**, and needs the tuned recipe first | 8 |
| `002_difficulty` | 18 | Tier-A difficulty screen at 1250 steps | 3.2.4 |
| `003_hardware_check` | 4 | H100/A100 run the flex compile path | — |
| `004_extended` | 20 | the screen at 2500 steps, matched pairs | 3.2.5 |
| `005_fgcount_ablation` | 3 | bias/question-node ablation — **result retracted** | 3.2.6 |
| `006_recipe` | 8 | **the recipe fix; `longest_chain` flips** | 3.2.7 |

Two things that are *not* here. The M2 canary's config was deleted (§3.2.3): it set
`node_position_mode`, which no longer exists, so it could only crash. The width analysis
(§3.2.5.1) has no config by design — it re-scores another sweep's checkpoints, and its per-run
inputs are that sweep's outputs; the reproduction procedure is written out in that section.

`results/` directories on disk still carry the **pre-renumbering** names (`001_smoke`,
`004_difficulty`, `006_extended`, `007_width`, `008_fgcount_ablation`, `009_recipe`) because they
were written before the renumber and are gitignored. Map them by name, not by number.

**Provenance of every number below.** Repo-internal results are cited to their file.
External anchors were read this session from the papers named inline. Anything marked
*(estimate)* is arithmetic, not measurement, and M0/M2 replace it.

---

## 0. The trap this plan is built to avoid

Molecule property prediction is the one domain where GTLM's headline mechanism is **void**.
The thesis (`CLAUDE_CONTEXT.md` §1) is that graph-tokenizing pipelines destroy node text by
compressing it to one vector, and GTLM wins by not compressing it. An atom is `C`. There is
no text to preserve. On `ogbg-molhiv`, GTLM is a graph transformer with a 1B LLM bolted on,
competing against Uni-Mol and GEM — models pretrained on ~10M molecules **with 3D
conformers**, which GTLM has no channel for.

Framed as "beat MoleculeNet SOTA", this campaign fails, expensively, and produces a
citable-against-us result. Same shape as the rel-trial outcome (`project-reltrial-headline-result`:
graph −7.7 pp, 11.8σ, bias channel inert), and `src/generalist/PLAN.md` §7 already writes the
mitigation for relbench — *"score it on its own terms"*. Molecules need the same discipline
written **before** the GPU commit, not after.

**So what are molecules actually for?** Four things, none of which is a leaderboard:

1. **Free, exact, unlimited structural labels.** RDKit is a ground-truth oracle. That makes
   molecules a *generator* domain, which §5 of the trunk plan prizes over corpora (single-pass,
   no overfitting, fresh eval sets, RLVR-eligible later per §11). No other real-data domain in
   the mixture has this property — KGQA, TAG and relbench are all finite corpora.
2. **Structural diversity the mixture does not otherwise have.** Every current domain is
   sparse-directed (KGQA Levi), ego-net (TAG), heterogeneous-bipartite (relbench) or abstract
   (GraphQA). Molecules are small, dense, undirected, cyclic, chemically typed, and
   *homogeneous in size*. D5 locks the trunk into the structural statistics of what it trains
   on; this widens them at near-zero cost.
3. **An honest flat twin.** SMILES is heavily represented in Llama's pretraining, so the
   flat-serialization control is a genuinely strong opponent here rather than a strawman. A
   graph-arm win over it means something.
4. **A property claim the leaderboard cannot touch** — permutation invariance over atoms
   (§6), which GTLM has by Property 1 and every SMILES LLM provably lacks.

**Target of the campaign, stated up front:** beat our own SMILES flat twin; land in the
InstructMol band at 1/7 the parameters; **do not claim to compete with Uni-Mol or any 3D
specialist.** Write that into the admission criterion the way §7 of the trunk plan demands.

---

## 1. Benchmarks — three tiers, in build order

### Tier A — the structural diagnostic suite (build first; it is free)

RDKit-generated QA over real molecules drawn from ZINC / ChEMBL / PubChem. This is the
`substructure` probe (`src/experiments/probes/README.md` Probe 2 — explicitly labelled
*"Proxy for: molecules"*) promoted from glued-together synthetic rings to real chemistry with
a real opponent.

| Task family | Question | Ground truth | Why it is here |
|---|---|---|---|
| `ring_membership` | "Is atom 14 part of a ring?" | `atom.IsInRing()` | the probe task, on real molecules |
| `ring_size` | "What is the smallest ring containing atom 14?" | SSSR | graded version of the above |
| `aromatic_ring` | "Is atom 14 in an aromatic ring?" | RDKit aromaticity | ring + chemical typing |
| `fg_presence` / `fg_count` | "How many carboxylic acid groups?" | SMARTS match | subgraph matching |
| `bond_path` | "How many bonds separate atom 3 and atom 14?" | graph SPD | SPD's home turf; direct read on the bias |
| `scaffold_share` | "Do these two molecules share a Murcko scaffold?" | RDKit scaffold | two-graph input; scaffold reasoning |
| `longest_chain` | "Longest unbranched carbon chain?" | graph traversal | global, not local |
| `stereo_potential` | "How many atoms *could* be stereocenters?" | `FindMolChiralCenters(includeUnassigned=True)` | pure connectivity, and hard — requires deciding whether four branches are constitutionally distinct |
| `stereo_assigned` | "Is atom 14 R or S?" / "How many stereocenters are *assigned*?" | CIP labels, which come from the SMILES `@`/`@@` tags | **deliberate negative control** — see below |

Metric: exact match / accuracy. 50/50 balanced where binary, as in the probe suite.

**This tier carries the scientific claim.** The repo's own evidence says it should be a rout:
on `substructure`, `none` 51.2 → `spd` 90.8 → `magnetic` 97.9
(`CLAUDE_CONTEXT.md` §4.4). And on `text_path`, the structural channel was worth ~36 points
*even though the task was fully solvable from the text alone* — the model does not reliably
extract topology from prose. SMILES is prose with ring-closure digits.

**The two stereo rows are a matched pair, and the distinction is the point.** Chirality is the
one chemical property that survives no 2D drawing: a carbon bonded to four *different* groups
has two mirror-image arrangements — same atoms, same bonds, same graph, different molecule
(and, for thalidomide, different pharmacology). SMILES carries it in the `@`/`@@` tags; a plain
atom-bond graph does not carry it at all unless it is written into the node text.

* `stereo_potential` asks *which atoms could be stereocenters* — determined entirely by
  connectivity, and genuinely hard (deciding four subtrees are pairwise non-isomorphic). The
  graph arm should **win**.
* `stereo_assigned` asks *which configuration each one actually has*. That information is in
  SMILES and provably not in our graph. The graph arm should **lose**.

A suite in which the graph arm wins everything cannot distinguish "the structural channel
works" from "something leaked" — a SMILES string in the prompt, a question generator with a
shortcut, a label correlated with molecule size. A control whose failure is *predicted in
advance, with a stated mechanism*, and which then fails for that mechanism, is what makes the
other rows trustworthy. Exactly the role the `direction` probe's construction plays
(`CLAUDE_CONTEXT.md` §4.4: symmetric features provably cannot separate the classes, so
`spd` 51.97 confirms the probe rather than disappointing).

**Should the stereo tags go into the node text?** Yes — for production, no — for the control,
and the two are a switch (`stereo_tags: on|off`), not a global decision. But *which* tag matters
more than whether:

* **The parity tag** (RDKit's `CHI_TETRAHEDRAL_CW`/`CCW` — the raw local handedness) **is
  standard input, not cheating.** OGB's own `atom_to_feature_vector` carries chirality as its
  second atom feature, so every GNN baseline on `ogbg-mol*` already receives it. Withholding it
  handicaps us against the baselines for no reason, and it costs real signal on Tier B, where
  binding and permeability are genuinely stereochemistry-dependent. **`stereo_tags: on` is the
  Tier-B and trunk default**, on the same fairness argument as §3.1 reason 1.
* **The CIP label** (`"chiral R"`) **would be cheating**, and the distinction is exactly the one
  that matters: parity is *information about the molecule*, the CIP label is *the answer to the
  task*. Never put it in the node text.

That distinction is what makes `stereo_assigned` a well-designed task rather than a lookup. The
parity tag is defined relative to the atom's own neighbour ordering; converting it to an R/S
answer requires ranking the four branches by CIP priority, which is a real computation over the
graph. So the task has three regimes, and running the first two is the measurement:

| `stereo_tags` | expected | what it proves |
|---|---|---|
| `off` | chance | the information genuinely is not in the graph — the negative control |
| `on` (parity) | high, but non-trivially so | node text reaches the model **and** the graph channel can do CIP-style ranking |
| CIP label in text | ~100% | nothing. Do not build this arm. |

Run `stereo_assigned` under both `off` and `on`; the **gap between them** is the result. That
converts the negative control into a positive test of information flow, which is strictly more
useful than a task we merely expect to lose. Record `stereo_tags` in every run record — a run
where it is silently `on` has no negative control left, and a Tier-B run where it is silently
`off` is quietly losing signal.

### Tier B — MoleculeNet (the field comparison)

Scaffold split (DeepChem's), ROC-AUC for classification, RMSE for regression.

Built and measured at M3 (`tier_b.py`). Molecule counts are post-drop; *examples* are
`(molecule, endpoint)` pairs after skipping absent labels:

| Dataset | molecules | endpoints | examples | positives | scaffold split | Role |
|---|---:|---:|---:|---:|---|---|
| BACE | 1513 | 1 | 1513 | 45.7% | 1210/151/152 | direct InstructMol anchor |
| BBBP | 2039 | 1 | 2039 | 76.5% | 1631/204/204 | direct InstructMol anchor |
| HIV | 41119 | 1 | 41119 | **3.5%** | 32895/4112/4112 | InstructMol anchor; the imbalance stress case |
| Tox21 | 7823 | 12 | 77864 | 7.5% | 63577/7139/7148 | multi-endpoint; 16012 labels absent |
| ClinTox | 1478 | 2 | 2956 | 50.6% | 2364/296/296 | **held out** (§4.1) |
| SIDER | 1427 | 27 | 38529 | 56.8% | 30807/3861/3861 | many endpoints, weak signal |
| ESOL / FreeSolv / Lipo | 1128 / 642 / 4200 | 1 | — | regression | — | needs `numeric_text`; `validate()` refuses them until it exists |

Two things this settles. **The multi-endpoint sets are the large ones** — Tox21 and SIDER
yield 78k and 39k examples from under 8k molecules each, so the "one example per
`(molecule, endpoint)`, endpoint named in the QUESTION node" decision is what makes them the
bulk of Tier B rather than a footnote, and it hands PLAN §4 arm 2 a large multi-task set for
free. And **HIV is the tie-collapse stress case**: ~145 positives in its test split, so a
handful of distinct margins would decide the AUROC outright. `n_distinct` and
`tied_pair_fraction` are in the run record from the first run for exactly this.

`ogbg-molhiv` optionally, if the OGB leaderboard framing is wanted; it is the same molecules
as HIV under a different split.

### Tier C — text-rich molecular language (optional in cycle 1)

**ChEBI-20 captioning** (33,010 molecule–caption pairs): graph in, free text out. The only
molecular task where the LLM half does real work and the output is not a token. Include it if
the "various types of problems" requirement needs a generative task; rank it third.

Caveat to write down before running: ChEBI-20 captions are highly templated and BLEU/ROUGE
reward template matching, so a strong number there is weak evidence. Do not let it carry the
claim.

### Deliberately excluded from cycle 1

* **QM9 / quantum properties.** They are 3D-determined. Without a 3D channel (§7) this is a
  guaranteed loss that says nothing about the architecture.
* **peptides-func / peptides-struct (LRGB).** Tempting for long-range, but *"Where Did the Gap
  Go? Reassessing the Long-Range Graph Benchmark"* showed most of LRGB's long-range gap
  dissolves under proper tuning. Keep peptides for the **size-generalization** protocol (§5),
  where they are a legitimate instrument, not as a long-range-superiority claim.
* **Molecule generation from text.** The output is a SMILES string; the graph is on the wrong
  side of the model. Nothing about GTLM is being tested.

---

## 2. Anchors, and what "beating them" means

Read this section the way `src/experiments/relbench/PLAN.md` §1 reads its own anchors: the
rows are not interchangeable and blending them produces a meaningless target.

**MoleculeNet classification, scaffold split, ROC-AUC ↑** (InstructMol, arXiv:2311.16208
Table 2; 3 seeds):

| | BACE | BBBP | HIV | what it is |
|---|---:|---:|---:|---|
| Uni-Mol (specialist) | 85.7 | 72.9 | 80.8 | **the ceiling, not the target** — 3D conformers, 10M-molecule pretraining |
| InstructMol-G (7B + graph tokens) | 84.3 | 68.6 | 74.0 | the graph-LLM band we should land in |
| InstructMol-GS (7B + graph + SMILES) | 82.1 | 72.4 | 68.9 | note it is *not* uniformly better than -G |
| Llama-2-7B-chat, LoRA | 74.8 | 65.6 | 62.3 | **the row to beat, at 1/7 the parameters** |
| Vicuna-7B, LoRA | 68.3 | 60.1 | 58.1 | |
| Galactica-6.7B | 58.4 | 53.5 | 72.2 | |

Mol-LLM (arXiv:2502.02810) reports BACE 80.5 and **BBBP 81.1** — the BBBP figure is far out of
line with every other row and its protocol differs (23-task generalist, SELFIES + hybrid
GINE/TokenGT encoder + Q-Former). Do not calibrate against it without reading its split.

**A caution on survey tables.** The 2026 systematic survey (arXiv:2604.16586) Table 4 quotes
SIDER 0.847 and ClinTox 0.984 from a KA-GAT/KA-GCN line of work. SIDER in the field usually
sits near 0.62–0.68. Those are scraped from source papers, not re-run. Calibrating against
them would set an unreachable target for reasons that have nothing to do with our model. The
same survey's own finding is the useful one: **method rankings are unstable across evaluation
protocols** — four different models win the six ADME endpoints under a time-based split.

**Molecule captioning, ChEBI-20:** MolT5-Large ≈ 25.9 BLEU-2; 3D-MolT5 42.05 BLEU-2 /
34.16 BLEU-4; MolCA (Galactica-1.3B) was SOTA at its time. All use 3D or large-scale
molecular pretraining we do not have.

### Pre-registered gates (write these into the config before the first run)

| Gate | Criterion |
|---|---|
| **A1** | Tier A: graph arm ≥ **+15 points** over the SMILES flat twin on `ring_membership`, `ring_size`, `bond_path`. (The probe gap was 51.2 → 97.9 against *no* structural channel; +15 against a real SMILES reader is conservative.) |
| **A2** | Tier A: graph arm **wins** `stereo_potential`; and on `stereo_assigned` sits at chance under `stereo_tags: off` while scoring well under `stereo_tags: on`. A graph arm above chance with tags `off` means something is leaking; find it before reading any other row. |
| **B1** | Tier B: graph arm beats **our own flat twin** by ≥ 1 sd on ≥ 4 of 6 classification sets. This is the real result. |
| **B2** | Tier B: within 3 AUROC of InstructMol-G on BACE/BBBP/HIV at 1B vs 7B. Aspirational, not a gate. |
| **B3** | Tier B: beat the Llama-2-7B-chat LoRA row (74.8 / 65.6 / 62.3) at 1B. If we do not, the flat twin will not either, and the domain is telling us the molecules are being read badly — check M1's round-trip test before blaming the architecture. |
| **P1** | §6: flat twin AUROC spread across randomized SMILES ≥ 2 points; GTLM spread < 0.1 (bounded by Property 1's 2.77e-5). |
| **Null** | If the graph arm does not beat flat on **Tier A**, the *encoding* is wrong, not the architecture — Tier A is structural by construction. Do not theorize about the model until the round-trip test and a `bias=none` control have both been run. Cf. `feedback-verify-nulls-are-real`: prove the bias modules left their init before reporting anything. |

---

## 3. Encoding — the decision this document exists to make

`src/generalist/PLAN.md` §7: *"needs bond types (D1) and a representation decision (SMILES vs.
atom-node text) that will matter more than most modelling choices."* Agreed. Here it is.

### 3.1 Default: `atom_levi_rich`

**Nodes = atoms. Node text = a short natural-language atom descriptor, never the bare symbol.**

```
"carbon aromatic ring deg3 H1"        rather than       "C"
"oxygen carbonyl deg1 H0"                               "O"
"nitrogen ring deg3 H0 charge+1"                        "N"
```

This is the single most consequential choice in the plan, for three independent reasons:

1. **Fairness.** It is exactly the information a GNN receives in its atom feature vector
   (element, degree, formal charge, hybridization, aromaticity, ring membership, implicit H).
   A bare-symbol encoding would be handicapping ourselves relative to every baseline.
2. **Cost.** `CLAUDE_CONTEXT.md` §4.5 measured overhead scaling as **nodes-per-token**: 7.9× at
   2048 nodes × 2 tokens vs 1.45× at 512 × 32. Bare atom symbols are the worst corner of that
   curve. Going from 1 to ~6 tokens/atom moves molecules ~6× toward the cheap end, at an N
   where the absolute cost is negligible anyway.
3. **It is what makes this GTLM and not a graph transformer.** The backbone can read
   "aromatic", "carbonyl", "chiral" as language it already knows. Bare symbols give it nothing
   to work with and reduce the LLM to an expensive random feature map.

**Bonds = Levi nodes** (D1 is decided: Levi). Text `"single bond"` / `"double bond"` /
`"triple bond"` / `"aromatic bond"`, `+ " in ring"`, `+ E/Z` where defined.

**Molecules are the cheapest domain in the mixture** — worth stating explicitly, because the
"many nodes, no text" intuition suggests the opposite. Measured sizes are in §3.1.1. Against
the ~1.05 s/it KGQA anchor at ~2k tokens (`src/generalist/PLAN.md` §8), expect **~0.2–0.4 s/it**
*(estimate — M2 measures it)*.

### 3.1.1 Measured at M0 (2026-08-28) — three corrections to this plan

`analyse_dataset.py`, all nine Tier-B datasets, Llama-3.2-1B tokenizer, 800-molecule samples
for token counts and the full corpus for sizes and splits.

| dataset | kind | mols | dropped (parse/bond) | endpoints | atoms mean/p95/max | Levi N mean/p95 | scaffold split |
|---|---|---:|---:|---:|---|---|---|
| `bace` | class | 1513 | 0/0 | 1 | 34.1 / 48 / 97 | 72.9 / 101 | 1210/151/152 |
| `bbbp` | class | 2039 | 11/0 | 1 | 24.2 / 39 / 101 | 52.3 / 84 | 1631/204/204 |
| `hiv` | class | 41119 | 7/1 | 1 | 25.6 / 47 / 116 | 55.2 / 98 | 32895/4112/4112 |
| `tox21` | class | 7823 | 8/0 | 12 | 18.7 / 38 / 124 | 40.2 / 81 | 6258/782/783 |
| `clintox` | class | 1478 | 4/2 | 2 | 26.2 / 55 / 136 | 56.0 / 117 | 1182/148/148 |
| `sider` | class | 1427 | 0/0 | 27 | 33.6 / 82 / **492** | 71.0 / 169 | 1141/143/143 |
| `esol` | regr | 1128 | 0/0 | 1 | 13.3 / 26 / 55 | 29.0 / 56 | 902/113/113 |
| `freesolv` | regr | 642 | 0/0 | 1 | 8.7 / 18 / 24 | 19.1 / 39 | 513/64/65 |
| `lipo` | regr | 4200 | 0/0 | 1 | 26.9 / 38 / 65 | 58.1 / 82 | 3360/420/420 |

| dataset | flat SMILES | `rich_levi` | `terse_levi` | `rich_atom_only` | nodes/token rich | nodes/token terse |
|---|---:|---:|---:|---:|---:|---:|
| `bace` | 42 | 368 | 113 | 354 | 0.20 | 0.65 |
| `bbbp` | 32 | 266 | 82 | 256 | 0.20 | 0.64 |
| `hiv` | 30 | 281 | 90 | 271 | 0.20 | 0.61 |
| `tox21` | 24 | 201 | 67 | 200 | 0.20 | 0.60 |
| `clintox` | 39 | 283 | 87 | 277 | 0.20 | 0.64 |
| `sider` | 46 | 342 | 106 | 343 | 0.21 | 0.67 |
| `esol` | 14 | 148 | 53 | 145 | 0.20 | 0.55 |
| `freesolv` | 9 | 98 | 40 | 100 | 0.19 | 0.48 |
| `lipo` | 31 | 305 | 96 | 286 | 0.19 | 0.61 |

**Correction 1 — the size estimate was right, the cost claim was overstated.** N ≈ 40–73 and
L ≈ 200–370 for the classification sets, close to the estimated (52, 250–400). But §3.1's claim
that rich text moves molecules "~6× toward the cheap end" is wrong: the measured nodes-per-token
is **0.20 rich vs 0.61 terse, a 3× reduction, not 6×**, because Levi bond nodes carry short text
in both arms and dilute the effect. The direction of the argument survives; the magnitude was
optimistic. For scale, the measured overhead curve is 7.9× at 1.0 nodes/token and 1.45× at 0.031
(`CLAUDE_CONTEXT.md` §4.5), so **both arms sit in its interior and neither is in the bad corner**.

**Correction 2 — SIDER has a 492-atom molecule** (~1000 Levi nodes, p95 169). Every other set is
tightly bounded. So `max_nodes` cannot be set from the mean, and SIDER alone determines whether
the D5 cap needs a policy here rather than just a number. It is also the set with 27 endpoints,
so it is simultaneously the widest and the most skewed — treat it as the domain's stress case.

**Correction 3 — a bond type we cannot encode exists, and it is negligible.** Across 1.63M bonds
in Tier B: 1,625,615 single/double/triple/aromatic and **10 dative**, all in organometallic iron
complexes in HIV and ClinTox. Dative bonds are directional and the Levi encoding is undirected,
so those 3 molecules are dropped and counted (`is_encodable`) rather than silently mis-encoded.
Parse failures are separately counted: 30 across the corpus, mostly BBBP's known-bad SMILES.

### 3.2 Arms to measure (all cheap at N ≈ 52)

**Two independent axes, which are easy to conflate.** Axis 1 is *how much text a node carries*.
Axis 2 is *whether a bond is a node at all*. They are orthogonal in intent but not quite in
practice (see the empty cell):

| | `levi` — bond becomes its own node | `atom_only` — no bond nodes |
|---|---|---|
| **`rich`** — `"carbon aromatic ring deg3 H1"`, `"double bond in ring"` | **the proposed default** | bond order folded into the atom's own text (`"deg3 arom2 single1"`) |
| **`terse`** — `"carbon"`, `"double"` | the cheap honest baseline — **run it** | **✗ never run.** Information-destroying: with no bond node *and* no bond summary in the atom text, a double bond is indistinguishable from a single one. Not a weak arm — an invalid one. |

**Three cells, not four**, and the fourth is discarded by construction rather than by
measurement. `terse × levi` is the one that matters most for the argument in §3.1: if it matches
`rich × levi`, the rich-featurization case is wrong and the featurizer code is unnecessary — a
cheap result either way, and worth knowing before M4 spends anything.

**Pre-registered prediction (DV, 2026-08-29, before M4 runs): `rich × levi` wins, and its margin
grows with task difficulty.** Written down in advance so the sweep can contradict it. Two things
would falsify it, and both are cheap to read off M4's table:

* `terse × levi` matching `rich × levi` on the hard families ⇒ the featurizer is dead weight and
  §3.1's whole argument for it is wrong.
* the `rich` margin *shrinking* rather than growing as the families get harder ⇒ rich text is
  helping the easy lookups (where the answer is nearly copied out of a node's own description)
  and not the multi-hop ones — the opposite of the stated mechanism, and a much more interesting
  result than confirmation.

Note §3.2.1's asymmetry cuts *for* the prediction on one specific family class: `terse` cannot
reconstruct aromaticity for ~6% of drug-like molecules (pyrrole-type N-heterocycles need the
hydrogen count), so `aromatic_ring` and the ring families have a mechanism by which `rich` should
win. That is a reason to check whether any `rich` win is confined to those families rather than
being the general effect the prediction claims.

### 3.1.2 Measured at M2 (job 134071, 2026-08-28) — flex vs eager, and the real cost

Six runs, 30 steps each, `ring_membership` on BACE, batch 4 × accum 8 (32 examples per
optimizer step). **No accuracy from this sweep is quotable** — see §8 M2/M3 — but the
speed and memory numbers are exactly what it was for.

| arm | bias | impl | ms/step | peak GB |
|---|---|---|---:|---:|
| flat | none | eager | **451** | 11.5 |
| flat | none | flex | 629 | 11.6 |
| graph | none | eager | **637** | 26.1 |
| graph | none | flex | 882 | 20.6 |
| graph | spd+magnetic | eager | 2177 | 26.1 |
| graph | spd+magnetic | flex | **1493** | **20.8** |

**Flex vs eager is settled, and the answer is conditional** — which is why §3.4 was right to
leave it open rather than assume. With the bias **on**, flex is **1.46× faster and uses 20%
less memory**, and it halves the bias overhead (eager 637→2177 = 3.4×; flex 882→1493 =
1.7×). With the bias **off**, eager wins on both arms: there is nothing for flex's
block-skipping to skip at N ≈ 59 (measured token sparsity 0.009) and its overhead dominates.
**Decision: flex for the graph arm, eager for the controls** — with the caveat that at 30
steps the ~320s autotune is not amortised, so `flex_compile_mode` matters more than the
ranking above suggests for short runs.

**The cost of the whole architecture on molecules: 4.8×** plain-LLM (flat eager 451 → graph
eager 2177), or **3.3×** using each arm's best backend. That is in line with the ~3×
published figure and confirms molecules are not in the bad corner of the overhead curve.
Per example: **47 ms** on the best graph configuration.

### 3.2.1 What each encoding actually preserves — measured at M1 (2026-08-28)

`roundtrip_check` encodes a molecule, rebuilds one **from the node text and topology alone**
(nothing is read back from the source molecule — the element comes from the atom node's word,
the bond order from the bond node's word), and compares. Over **every molecule in Tier B, all
three encodings — 61,369 × 3 round trips — zero failures.**

That took four iterations, and each failure was a genuine finding rather than a typo:

| What failed | Cause | Resolution |
|---|---|---|
| `rich_levi`, ~1% | isotopes (`[99Tc]` tracers) and radical electrons (`[N]=O`) were **not in the node text** | added `iso<n>` / `rad<n>` fields — a real gap in the "rich atom features" spec |
| `rich_levi`, 1 molecule | an explicit `[H]` *atom* alongside its parent's own H count, double-counted | `RemoveAllHs` at load. Not `RemoveHs`, which deliberately keeps an H that defines bond stereo — which was exactly this molecule |
| `terse_levi`, 6.3% | comparing as SMILES asks terse for something it cannot give — see below | comparison level corrected to `labelled_graph` |
| `terse_levi`, the rest | `SanitizeMol` **kekulises**: it rewrote as single/double the aromatic bonds the encoding had carried correctly | compare the unsanitised rebuild; the mutation was in the decoder, not the encoding |

**The terse finding is the substantive one, and it is a prior for the `terse × levi` arm.**
Aromaticity perception requires hydrogen counts: pyrrole's `[nH]` is aromatic *because* that
nitrogen carries an H. `terse` emits "nitrogen" and drops the count, so the ring cannot be
re-perceived and RDKit kekulises it into a saturated one. The labelled graph survives intact —
elements, bonds and bond orders all round-trip exactly — but **the molecule is no longer
reconstructible as a chemical object**, for 6.3% of drug-like molecules, concentrated in
N-heterocycles, among the most common motifs in medicinal chemistry.

So `terse` is not merely "less verbose than rich". It is lossy in a specific, chemically
important place. That does not remove it from the sweep — it may still match `rich` on tasks
that do not need aromaticity, and M4 should still run it first because it is the cheapest
run — but it does mean a `terse` loss on Tier B should be attributed to *this*, and checked
against the aromatic subset, before it is read as evidence about node-text richness in general.

### 3.2.2 The M2 accuracy anomaly, and what was ruled out

001 reported graph-arm exact match of **0.000** (flat: 0.540, which is exactly the label
prior — the flat arm learned "always Yes" immediately) with graph loss starting at **13.48**,
*above* uniform over Llama's 128k vocab (ln 128256 = 11.76). A model that starts worse than
uniform is confidently predicting the wrong token, which normally means misalignment.

**Ruled out, in order:**

* **The data.** The prompt node tokenizes to `["\n", "A", ":", " No"]` with labels
  `[-100, -100, -100, 2360]` — the answer is the last token and it is the only supervised
  one. Verified on the built artifact, both arms.
* **The collation.** `GraphCollatorV2._pack_one` packs the prompt node last *by index*
  regardless of where RCM moved it, and places labels at
  `[prefix_len : prefix_len + prompt_len]` after asserting that length matches the prompt
  node's. The RCM-moves-the-prompt-node theory is wrong.
* **The bias.** `bias=none` shows the identical 0.000 / 13.48, so the graph bias is not
  implicated.
* **The comparison to `probes`** (initial loss ~5.1, accuracy at chance from step 0) — this
  is the *wrong reference*. Probes puts the question **inside** the prompt node, so its
  answer position follows a complete question at contiguous RoPE positions. Ours does not.

**The right reference is graphqa's `question_node: isolated` arm**, whose
`ANSWER_PREFIX = "A:"` produces a prompt node of `"A: <answer>"` — structurally identical to
ours, and worth +2.1pp on GraphQA. So the construction is proven elsewhere.

**What the model is actually doing — measured, not inferred.** A CPU forward pass of the
untrained model over both arms, reading the top-5 at the answer position:

| arm | context | top-5 predictions |
|---|---|---|
| flat | `…H:25]1\nA:` | `' True'`, `' '`, `' yes'`, `' Yes'`, `' true'` |
| graph | `… carbon deg1 H3\nA:` | `' ring'`, `' deg'`, `' carbon'`, `'9'`, `'1'` |

The flat arm knows a yes/no answer is due (loss 3.2–4.0). **The graph arm is continuing the
atom description** (loss 8.9–16.7). It is not confused about the question; it does not
believe a question is being asked at all.

**The mechanism is positional, and it was verified directly.** Under
`node_position_mode="reset"` the prompt node's `"\nA:"` sits at RoPE positions **0, 1, 2 —
the same positions that begin every atom node** — while prefix nodes reach only position 10.
Nothing distinguishes "the answer anchor" from "another node is starting", so the model
continues the sequence's dominant pattern, which with rich atom text is atom description.
Switching to `spd_depth` moves the prompt node to positions **363–366 against a maximum
prefix position of 362**, i.e. strictly clear of the prefix:

```
reset      prompt positions [0, 1, 2, 3]        max prefix position 10
spd_depth  prompt positions [363, 364, 365, 366] max prefix position 362
```

**This is the documented RoPE-shock hypothesis** (`src/models/TODO.md`;
`src/generalist/PLAN.md` D4: *"per-node position resets feed the frozen backbone an
out-of-distribution positional signal"*), and it explains why GraphQA survives `reset` while
molecules do not: **GraphQA's node texts are bare integers**, so there is no repeating
textual pattern for the model to continue. Rich atom text creates a strong competing
continuation prior that three tokens at position 0–2 cannot override. That makes this a
finding about **text-rich GTLM domains in general**, not a molecules workaround.

> **Struck 2026-08-29.** The paragraph above originally ended "*and it is a direct argument for
> `spd_depth` as a trunk default wherever node text is substantive*." That conclusion was wrong
> twice over: the canary (§3.2.3) showed there was nothing to fix, and kgqa had **already
> measured `spd_depth` and rejected it** at −9.4 F1 (E3, `kgqa/README.md`) — a result that was in
> the repo the whole time and that this section should have cited before proposing the knob as a
> trunk default. The positional measurements below are kept because they are correct; the
> recommendation built on them is withdrawn.

### 3.2.3 RESOLVED at M2-canary (job 134111, 2026-08-29): nothing was broken

The canary config ran the two candidate fixes against the unfixed baseline at 1250 steps.
(**That config has since been deleted.** It set `node_position_mode`, which no longer exists as
a `RunConfig` field, so it could only crash — and a config that cannot run is worse than no
config, because it invites someone to try. Its evidence is this section; nothing else depended
on it. Every config that remains is verified to expand and construct.)
Result, `ring_membership`, one seed:

| arm | `node_position_mode` | `question_node` | test acc | bias norm init → final |
|---|---|---|---:|---|
| graph | `reset` | `isolated` | **1.000** | 6.27 → 23.9 |
| graph | `reset` | `off` | 1.000 | 6.27 → 23.9 |
| graph | `spd_depth` | `isolated` | 0.998 | 6.27 → 18.7 |
| **flat** | `reset` | `isolated` | **0.877** | 0.0 → 0.0 (no bias exists) |

(`node_position_mode` appears in this table because the canary ran it. It is no longer a knob of
this experiment — see the decision below.)

**The unfixed baseline solves the task perfectly. Neither fix was needed.** The 0.000 at 30
steps was undertraining, full stop, and `feedback-dont-call-floors-early` was the applicable
rule the whole time.

**What survives from §3.2.2 and what does not.** The *mechanism* is real and measured: an
untrained model does continue the atom description, and `reset` does place the prompt node at
positions 0–2 inside the prefix's range. What does **not** follow is that this needs fixing —
1250 steps of LoRA is more than enough to override the continuation prior. Read §3.2.2 as a
description of the initialisation, not as a defect report.

**DECISION 2026-08-29 (DV): `node_position_mode` is unwired from this experiment entirely.**
Not defaulted off — removed. There is no `RunConfig` field, no `--node-position-mode` flag, and
no entry in the run record, and `train.py` is back on the shared
`expressiveness/training/dispatch.build_collator` (the local `GraphCollatorV2` construction
existed *only* to pass this knob). A test asserts the field cannot be set at all.

The reasoning, which is about evidence rather than tidiness. `spd_depth` now has **two
measurements and no positive result**:

| where | `reset` | `spd_depth` | |
|---|---:|---:|---|
| kgqa E3, WebQSP, 3 seeds (`kgqa/README.md`, `032_node_position_spd_depth.jsonc`) | 0.7351 ± 0.0076 F1 | **0.6412 ± 0.0037** | −9.4 F1, fix rejected |
| molecules canary, `ring_membership`, 1 seed | **1.000** | 0.998 | lower, and less bias movement (18.7 vs 23.9) |

A knob that is wired but never set is worse than no knob: a sweep can put it in the axis list,
the run record reports it faithfully, and the result is a campaign arm that was never justified.
`GraphCollatorV2` still implements `spd_depth` and kgqa still owns it as a documented negative
result — that is the right home for it. If it is ever revisited it should be as a deliberate
experiment there, not as an inherited default here.

**`question_node` stays, and stays on.** It is settled rather than open: the question lives in
the **prefix**, in its own edge-free node, so every atom and bond node attends to it and node
representations are question-conditioned. Edge-free is the point — the question is visible
through attention but contributes nothing to the SPD / magnetic features, so the structural bias
still describes the molecule alone. With `off` the prefix is query-blind and the molecule must be
encoded question-agnostically (the layout `probes` uses, retained only so it stays expressible).
Do not move it without a concrete reason. The canary's 1.000 vs 1.000 does **not** choose between
them; `ring_membership` is saturated and cannot choose between anything.

**Renamed 2026-08-29 (DV): the values are `"on"` / `"off"`, not `"isolated"` / `"off"`.** graphqa
and kgqa keep `"isolated"` because there the placement is one of several conceivable ones; here
only one is worth having, so the value is named for what it does. `"isolated"` is **rejected**,
not aliased — a silently-accepted synonym would put two spellings of one arm into the run
records. Two consequences worth knowing when reading results:

* Every table above that says `isolated` means today's `on`. The canary predates the rename.
* **`002_difficulty`'s `runs.jsonl` straddles it**: tasks that started before the rename record
  `question_node: "isolated"`, later ones record `"on"`. Same layout, same graph, two spellings —
  group them together. The dataset cache is unaffected, because `dataset_path` tags the key only
  when the value differs from the default, so a rename of the default changes no path (pinned by
  `test_default_question_node_leaves_the_cache_path_untagged`).

**The `002_difficulty` screen is unaffected and did not need re-running.** Its job scripts never passed
either flag, so all 18 runs use the question node on and reset positions. The collator swap
is a verified no-op, not an assumed one: `_node_offsets` returns `{j: 0}` at
`text_graph_collator_v2.py:330` under `reset`, before `max_spd` — the only argument that
differed — is ever read. Tasks that started on the old code and tasks that start on the new one
are directly comparable.

**The real finding is the ceiling.** All three graph arms hit ~1.000, so `ring_membership` is
**saturated and cannot discriminate between encodings, bias arms, or anything else** — every
variant would return 1.000 and the sweep would be a table of ties.

The flat arm's 0.877 settles what that ceiling means. The graph channel is worth **+12.3
points** here, which is a real gap but **short of gate A1's +15**, and short of it for an
uninformative reason: the graph arm is clamped at 1.000, so the measured gap is bounded by
flat's headroom rather than by anything about the graph. **Gate A1 is therefore neither passed
nor failed on `ring_membership` — the task cannot evaluate it**, and the gate must be judged on
a family where the graph arm is off the ceiling. Recording it as a near-miss would be reading a
truncated measurement as a substantive one.

Consequence for the build order: **M4's encoding sweep must not run on `ring_membership`**, and
task selection stops being an assumption and becomes a measurement — hence `002_difficulty`
(§3.2.4), a screen over all nine runnable Tier-A families before any encoding is compared.

### 3.2.4 M2b — the Tier-A difficulty screen (job 134202, submitted 2026-08-29)

`ring_membership` saturating is not a fact about that one family; it is a warning that Tier A
was designed by intuition and never measured. Nine of the ten families are runnable (`bond_path`
is held out and `load_data` refuses it), and any of them could be at ceiling, at chance, or
measuring nothing but the LLM's text handling. `002_difficulty.jsonc` runs all nine × both arms
= **18 runs** under the canary's exact recipe, so its `ring_membership` cell reproduces 1.000 / 0.877
as a within-sweep control.

Admission criteria for entering the M4 encoding sweep, fixed before the results land:

1. **Headroom** — graph accuracy ≤ 0.95. At ceiling nothing can be ranked.
2. **Signal** — graph − flat ≥ 0.03. Below that the family is measuring the LLM's handling of
   SMILES text, not the graph channel, and an encoding change has nothing to move.
3. **Learnability** — graph accuracy well above the majority-class rate. A family at the base
   rate is uninformative about encodings too.

A family failing (1) is still *reportable* as a win; it is just useless as an instrument. A
family failing (3) gets a longer budget before any verdict (`feedback-dont-call-floors-early`),
not a drop. `stereo_assigned` is in the screen precisely because it is the designed-in loss —
chirality reaches the graph only as a parity tag, so flat winning or tying there is the suite
confirming it measures what it claims (§5).

Gate A1 will be judged on the surviving families, not on `ring_membership`.

#### 3.2.4.1 The screen's own confound, found mid-flight (DV, 2026-08-29)

Two of the first completed pairs went to the flat arm on test accuracy. Reading that as
"flat wins" was wrong, for a reason worth writing down because it will recur in every
cross-modality comparison this project runs.

**Accuracy at the first eval is not a measurement.** `fg_count`'s answer distribution is
76.0% `" 0"`. Both arms score 0.740 / 0.742 at eval 1 — i.e. **both are predicting the mode
and have learned nothing**. Concluding from that that "the modality shift looks cheap" (as
this plan briefly did) compares two constants.

**What the curves actually show**, validation accuracy every 100 steps:

```
fg_count   (base rate 0.760)
  graph    0.740 0.742 0.742 0.752 0.748 | 0.778 0.800 0.784 0.798 0.816 0.820 0.814
  flat     0.742 0.766 0.778 0.778 0.808 | 0.828 0.832 0.840 0.850 0.852 0.852 0.852

stereo_potential   (base rate 0.285)
  graph    0.376 0.536 0.594 0.654 0.708 0.716 0.768 0.740 0.776 0.770 0.780 0.806
  flat     0.348 0.538 0.530 0.636 0.672 0.704 0.738 0.744 0.762 0.762 0.756 0.766
```

On `fg_count` the graph arm is **pinned at the base rate for 500 steps** while the flat arm
departs it by step 200, then the graph arm climbs steeply and is **still climbing when the
cosine schedule anneals the LR to min**. That is a *start-up cost*, not a lower ceiling: the
model is being handed a modality it has never seen, where flat SMILES is something Llama
already read during pretraining. Its 0.809 is a **lower bound**.

On `stereo_potential` the graph arm leads at 11 of 12 eval points and ends 0.806 vs 0.766 on
validation with a much lower eval loss (0.530 vs 0.677) — yet **loses on test**, 0.779 vs
0.805. Same splits, same seed, opposite sign. One seed cannot support either direction.

**Three consequences, all now enforced in code rather than remembered:**

1. **`base_rate`, `answer_distribution` and `n_classes` go in every run record**
   (`train.py::_answer_stats`, read from the `.gtds` sidecar so a warm cache still reports
   it). Criterion 3 was unverifiable without it.
2. **`eval_curve`, `tail_gain` and `still_improving` go in every run record**
   (`train.py::_convergence`). An arm whose metric gains >1pp over the last three evals is
   **budget-limited**, and its headline is a lower bound; the run log now says so in words.
   Comparing a converged arm against an interrupted one is not a comparison.
3. **Admission (and every gate) is judged on validation curves plus convergence state, not
   on a single test scalar at one seed.** A budget-limited arm gets a longer run before any
   verdict — `feedback-dont-call-floors-early` applied to a ceiling instead of a floor.

**Also flagged: `fg_count` and `fg_presence` are weak instruments by construction** — both
have a 0.760 base rate, so the whole usable range is 24 points wide and accuracy is a poor
metric in it. `stereo_potential` (0.285) is much better shaped. Base rate belongs in the
admission decision alongside headroom and signal.

#### 3.2.4.2 Reporting convention for mixed-budget results — REQUIRED (DV, 2026-08-29)

Once some cells are re-run at a longer budget, a results table silently mixes two budgets.
That is the kind of thing a reader cannot detect and an author forgets. So:

**Every cell measured at the extended budget carries a `*`, and every table containing one
carries this note beneath it:**

> `*` re-run at double the budget (2500 steps vs 1250). A cell was re-run when its arm was
> **still improving** at the end of the `002_difficulty` run — validation metric gaining more than 1pp
> over the last three evals (`still_improving` in the run record). Both arms of an affected
> pair were re-run, including an already-converged one, so that every graph-vs-flat
> comparison is between two runs at the same budget. Cells without a `*` converged at 1250
> steps and were not re-run.

**Superseded in practice by §3.2.7.** Once `004_extended` re-ran *every* cell, the budget stopped being
mixed and the `*` had nothing to mark: §3.2.5's table is uniformly 2500 steps and §3.2.7's is
uniformly 5000. Keep the convention for any future table that does mix budgets, and keep the
reasoning behind it — re-run the pair, and trigger on the curve rather than the score — which is
what makes those two tables comparable at all. But note that §3.2.7 also **falsified the trigger**:
`still_improving` marked a cell converged that then gained 10 points. The convention is sound;
the detector under it is not sensitive enough to justify a plateau claim on its own.

Two rules behind that note, both deliberate:

1. **Re-run the pair, not the cell.** If only the climbing arm were re-extended, every
   starred comparison would pit a 2500-step run against a 1250-step one — replacing a bias
   against the graph arm with a bias in its favour. Re-running the converged arm too costs
   one job and buys a matched comparison; it also *tests* the claim that it had converged,
   since a converged arm given double the budget should return the same number.
2. **The trigger is the convergence flag, not the score.** Re-running the cells that lost
   would be selecting on the outcome. `still_improving` is a property of the curve, computed
   the same way for both arms, and it flagged `ring_membership`'s *flat* arm — a cell where
   the graph arm was already winning — as readily as it flagged the graph arm's losses.

**Atom referencing (`atom_labels`).** Tier-A questions name an atom ("is atom 14 in a ring?")
and nothing in either arm named one. Both arms now do: `atom14` prefixed to the node's text,
`[cH:14]` atom-map numbers in the flat SMILES. Labels are **1-based in both**, because RDKit
reads map number 0 as "unmapped" and a 0-based scheme drops atom 0's label *in the flat arm
only* — silently, and in the direction that flatters the graph arm. Pinned by a test.

The other three axes:

| Axis | Arms | What it decides |
|---|---|---|
| sequence | `graph_only` / `+smiles` | SMILES in the QUESTION node |
| bias | `none` / `spd` / `spd+magnetic` / `spd+magnetic_shared` | mirrors the probe suite arms |
| serialization control | **flat twin** (canonical SMILES, same prompt, same recipe) | mandatory per trunk plan §3.3 |

**`atom_only` is not a formality.** Levi doubles N, doubles every shortest path, and turns a
benzene 6-ring into a 12-cycle. The `substructure` probe says the *spectrum* is what carries
ring detection (`magnetic` 97.9) — and Levi changes that spectrum. The control folds bond order
into the atom's own text (`"deg3 arom2 single1"`) and keeps the atom-atom topology intact. If
`atom_only` wins on Tier A, that is a finding worth reporting back into D1: Levi is right for
*text-bearing* edges (KGQA relations carry sentences) and possibly wrong for *typed* edges with
a four-symbol vocabulary.

**`+smiles` is an arm, never the default.** InstructMol-GS vs -G is +3.8 on BBBP and −5.1 on
HIV — genuinely task-dependent. The headline must be `graph_only`, or the claim is unclean in
exactly the way the WebQSP triplet arm is (`src/generalist/PLAN.md` §3.2: *"must not
reintroduce that encoding under another name"*). A SMILES string in the prompt is a flat
serialization sitting inside a graph arm.

### 3.2.5 M2c — the screen at double budget (`004_extended`, 2026-08-29), and what it admitted

All 18 cells of `002_difficulty` re-run at 2500 steps under §3.2.4.2's pair rule, plus `fg_atom_membership`
(added as the atom-level twin of `fg_presence`, §3.2.6). Twenty runs. Test accuracy, one seed:

| family | base rate | graph | flat | graph − flat |
|---|---:|---:|---:|---:|
| `aromatic_ring` | 0.511 | 1.000 | 0.997 | +0.003 |
| `ring_membership` | 0.505 | 1.000 | 0.941 | +0.059 |
| `stereo_assigned` | 0.715 | 0.997 | 0.996 | +0.001 |
| `ring_count` | 0.326 | 0.993 | 0.993 | 0.000 |
| `fg_atom_membership` | 0.503 | 0.913 | 0.938 | −0.025 |
| `ring_size` | 0.498 | 0.901 | 0.827 | **+0.074** |
| `stereo_potential` | 0.285 | 0.877 | 0.907 | −0.030 |
| `longest_chain` | 0.253 | 0.855 | 0.934 | −0.079 |
| `fg_presence` | 0.760 | 0.849 | 0.902 | −0.053 |
| `fg_count` | 0.760 | 0.820 | 0.907 | −0.087 |

**Applied literally, §3.2.4's criteria admit one family.** Headroom (≤0.95) removes the top four;
of the six that survive it, only `ring_size` also clears signal (≥0.03), because the other five
have the graph arm *behind*. That is not a shortlist, it is the screen reporting that criterion 2
was written on an assumption — that where there is headroom the graph arm leads. **Do not weaken
the criterion to fit the result.** The right reading is that the instrument found something the
plan did not predict, and §3.2.6/§3.2.7 are the diagnosis, not a rescue.

**The split has a shape.** The graph arm wins or ties every family whose answer is local to a
named atom or a small neighbourhood (`ring_membership` +0.059, `ring_size` +0.074,
`aromatic_ring`, `stereo_assigned`) and loses every family requiring an aggregate over the whole
molecule (`fg_count` −0.087, `fg_presence` −0.053, `longest_chain` −0.079). `fg_atom_membership`
exists to test exactly that boundary: it is `fg_presence`'s question ("is a nitro group here?")
re-asked about one *named atom*. In skill above base rate the twin cuts the deficit from −0.221
to −0.050 — most of the gap is about *global aggregation*, not about functional groups.

**`stereo_potential` is not the designed-in loss; `stereo_assigned` is.** `stereo_potential` asks
whether a carbon *could* be a stereocentre, which is pure connectivity and the graph should win it
(gate A2). `stereo_assigned` asks for the assigned R/S tag, which reaches the graph only as a
parity tag and is the family where flat winning confirms the suite measures what it claims. Both
tie or narrowly favour flat here; neither is decidable at the stale recipe (§3.2.7).

### 3.2.5.1 `max_spd` is not the limitation — measured, not assumed (the width analysis, 2026-08-29)

Levi doubles every distance, so "the graph arm fails on molecules too wide for the clamp" is the
obvious hypothesis. It is also exactly the shape of hypothesis that produced `spd_depth`
(§3.2.3), so it was measured before anything was changed. `evaluate_checkpoint.py` re-scores the
18 existing `002_difficulty` checkpoints and writes per-example geometry (`analysis.py`); no
retraining, and it asks the question about *those* models rather than different ones.

**This one has no config, deliberately.** It is `--mode eval` over another sweep's checkpoints,
and both of its per-run inputs — the checkpoint directory and the `--expect-accuracy` the reload
is verified against — are *outputs* of that sweep, not values anyone can write down in advance.
A config would encode paths under gitignored `checkpoints/` and accuracies from one particular
execution of `002_difficulty`, and would fire spurious reload mismatches against any re-run of
it. To reproduce: run `002_difficulty`, then for each of its 18 checkpoints call
`python -m src.experiments.molecules --mode eval --checkpoint <dir> --expect-accuracy <the
test_accuracy that run recorded>` with `002_difficulty`'s data flags unchanged — they must match
exactly or `load_data` builds a different test split and the per-example rows describe examples
the model never saw.

* **Reach.** Levi diameter p50 = 26, p90 = 37, max = 76. About 26–30% of examples have at least
  one pair at or beyond `max_spd = 32` — but the **mean share of pairs clamped is 0.66–0.84%**.
  The clamp touches a quarter of molecules and, within them, under 1% of the distance matrix.
* **Shape.** If the clamp bound accuracy there must be a **step** at 32. There is none. On
  `longest_chain` accuracy *rises* across it (d≥22: 0.734, d≥28: 0.794, **d≥32: 0.826**);
  `ring_size` is flat throughout (0.904 / 0.887 / 0.906); `fg_count` dips at 32 and recovers at
  40. That is noise around a mild width effect the flat arm shows too, not a clamp.

**`max_spd` stays 32.** Raising it is cheap, which is precisely why it needed a reason: a cheap
change adopted without one is how a knob with no positive result gets into a sweep axis.

**The reload check fired, and the pattern exonerates the load path.** Six of 18 re-scores drifted
past the 0.005 tolerance (max 0.014). Drift is concentrated in the **flat** arm — five of the six
— which has *no bias parameters to lose*. This is bf16 eval nondeterminism under a different
batch composition, not `project-load-best-model-bias-bug` recurring. Recorded because the
opposite pattern (graph-only drift) would have invalidated every geometry number above.

### 3.2.5.2 Prompt edges are directed, verified — the hub does not shortcut the molecule

A prompt node wired to every atom with **undirected** edges would put all atoms at distance 2 of
each other and destroy the geometry the bias exists to describe. It would also make `magnetic`
degenerate for a second, unrelated reason. Checked directly on a cached graph rather than
reasoned about: the prompt node has **out-degree 13, in-degree 0**, and removing it changes
**zero** atom-atom shortest paths. The graph is built on `nx.DiGraph` and `attach_question` only
ever calls `add_edge(prompt_node, target)`, so this was already correct.

Recorded because the first diagnostic run at this question called `to_undirected()` before
measuring and therefore answered a question about a different graph. The measurement above
replaces it.

### 3.2.6 `005_fgcount_ablation` — and why its result is now INVALID

`fg_count` is the molecules analogue of GraphQA's `edge_count`: count occurrences across the
whole graph, no atom named. GraphQA `003_ablation` says the structural bias is load-bearing for
exactly those graph-level tasks (removing magnetic costs 54 points on `edge_count`, 11 on
`triangle_counting`) while every named-node task is unaffected. Molecules showed the opposite
split (§3.2.5), so this sweep asked which. Three arms at 2500 steps against the 0.820 reference:

| arm | test acc |
|---|---:|
| `spd+magnetic`, question node on *(`004_extended` reference)* | 0.820 |
| `spd` (no magnetic) | 0.822 |
| `none` (no structural channel at all) | 0.823 |
| `spd+magnetic`, question node **off** | 0.798 |

Read at the time: the bias contributes **nothing** on this task, so the deficit is about the
encoding competing with a pretrained-native SMILES string rather than about the wiring.

**That reading is retracted. This sweep ran at `bias_lr = 1e-3`, the lowest in the repo**
(§3.2.7). `project-landmark-campaign` records `bias_lr` as *the magnitude knob*, so "the bias
does nothing" and "the bias never reached useful magnitude" are indistinguishable here, and `006_recipe`
showed the second is live. **Do not cite the 0.820 / 0.822 / 0.823 null.** Retiring it properly
needs one run: `fg_count`, graph, `bias=none`, at the tuned recipe and 5000 steps. If it lands
near 0.897 the bias really is inert and `006_recipe`'s gain was adapter capacity; if it lands near 0.841
the bias became load-bearing when it was allowed to.

The question-node arm survives as a direction (`off` is worse, −0.022, consistent with §3.2.3's
argument that the prefix question makes node representations question-conditioned), but its
magnitude is measured at the same stale recipe and should not be quoted precisely.

### 3.2.7 `006_recipe` — the recipe was the blocker. `longest_chain` flips. (2026-08-29)

Molecules inherited `lr=1e-5, bias_lr=1e-3, lora_r=8` from `expressiveness`, the oldest
experiment in the repo, and never revisited it. Measured across every campaign's run records that
is the **lowest-capacity recipe anywhere**:

```
relbench          lr 2e-4   bias_lr 5e-2   r 32
kgqa / context    lr 1e-4   bias_lr 5e-3   r 64
probes            lr 5e-5   bias_lr 5e-3   r 16
graphqa           lr 3e-5   bias_lr 5e-3   r 16     (329 runs)
bias_experiments  lr 3e-5+  bias_lr 5e-3+  r 16-64
molecules         lr 1e-5   bias_lr 1e-3   r 8      <-- 3-20x / 5-50x below
```

Eight runs, 5000 steps (double `004_extended`, quadruple `002_difficulty`): two tasks × two arms
× `current` (the `002`/`004` recipe) and `tuned` (graphqa's — the closest comparable, and the one with the most runs behind
it, rather than newly invented values). **Both arms get the identical change**; tuning only the
graph arm would replace a comparison biased against GTLM with one biased in its favour. Running
`current` at 5000 as well separates *more steps* from *more capacity*, which a tuned-only sweep
would confound.

| family | arm | 1250 (`002`) | 2500 (`004`) | 5000 `current` | 5000 `tuned` |
|---|---|---:|---:|---:|---:|
| `longest_chain` (base 0.253) | flat | 0.844 | 0.934 | 0.948 | 0.947 |
| | **graph** | 0.778 | 0.855 | **0.959** | **0.989** |
| | *gap* | *−0.066* | *−0.079* | *+0.011* | ***+0.042*** |
| `fg_count` (base 0.760) | flat | 0.848 | 0.907 | 0.912 | 0.936 |
| | **graph** | 0.809 | 0.820 | 0.841 | 0.897 |
| | *gap* | *−0.039* | *−0.087* | *−0.071* | *−0.039* |

**Four findings, in order of how much they change the plan.**

1. **`longest_chain` flips sign: graph 0.989 vs flat 0.947.** Against a 0.253 base rate that is
   98.5% of available headroom against 92.9%, and 11 test errors in 1000. This is the **first
   molecule-level (not named-node) win in the domain**, and it removes §3.2.5's clean
   local-vs-global story as a statement about GTLM: the global family that was losing by 0.079
   now wins by 0.042, with no change to the encoding, the wiring, or the task.
2. **The recipe change is graph-specific on that task.** It moved flat by **−0.001** and graph by
   **+0.030** — same change, same data, same duration. `bias_lr` and adapter rank are the only
   knobs in it that touch the graph arm alone, which is consistent with `bias_lr` being the
   magnitude knob and directly implicates §3.2.6's null.
3. **Undertraining was a factor after all, and `still_improving` under-called it.** The detector
   (`tail_gain > 0.01`) marked `longest_chain` graph **converged** at 2500 steps (0.855); it then
   gained **+0.104** at the same settings. §3.2.4.1 installed that flag to stop exactly this
   error and it was not sensitive enough. Recurrence of `feedback-dont-call-floors-early`, now
   twice in this campaign (M2's floor, M2c's ceiling). Treat `still_improving = False` as weak
   evidence, never as licence to conclude a plateau.
4. **`fg_count` is a genuinely different case.** The gap halves (−0.087 → −0.039) but does not
   close, and **both `current` arms are still improving at 5000 steps** — those two cells are
   lower bounds, not plateaus. Counting occurrences across a whole molecule remains harder for
   this encoding than tracing a path through it.

**Consequences for the rest of the plan.**

* **The tuned recipe is the recipe.** M3, M4 and everything after run at `lr=3e-5, bias_lr=5e-3,
  lora_r=16`. Every number in §3.2.4–§3.2.6 is at the stale one and is a lower bound for the
  graph arm — cite them as such or re-measure.
* **The `002`/`004` screen must be re-read, not re-used.** Admission (§3.2.4) ranked families under a
  recipe that suppressed the graph arm by up to 13 points. A cheap re-screen at the tuned recipe
  on the six non-saturated families is the honest input to M4, and is ~12 GPU-h.
* **Gate A2 becomes decidable.** `stereo_potential` (pure connectivity, graph should win) sat at
  −0.030 under the stale recipe; that is not a verdict on the gate.
* The `bias=none` control in §3.2.6 is the single highest-value next run.

### 3.3 The bias channel behaves differently here — know this before reading the ablation

**Molecules are undirected, so the magnetic Laplacian's Hermitian phase is identically zero and
`magnetic` degenerates to plain-Laplacian spectral information.** This is stated in the repo:
`probes/README.md`, Probe 2 — *"on this undirected probe `magnetic` degenerates to plain
Laplacian information — confirming that is itself a result."*

Three consequences:

* The `direction` probe's headline (*magnetic is the only channel carrying edge direction*,
  `CLAUDE_CONTEXT.md` §4.4) **does not transfer**. On molecules magnetic is a spectral channel.
  Anyone reading a molecule ablation next to the KGQA one will assume otherwise unless it is
  written down.
* `magnetic` should still be load-bearing here — cycle detection is precisely where spectral
  features classically win, and the probe measured 97.9 vs 90.8 for SPD. **Still untested.**
  `005_fgcount_ablation` (§3.2.6) appeared to refute it and has been retracted: it ran at the stale `bias_lr` where
  an inert bias and an under-driven one look identical. The claim is open, and the run that
  settles it is `fg_count` × `bias=none` at the tuned recipe.
* D2's cost proviso (`project-probe-suite`: rule picks magnetic k4, cost proviso triggered)
  is nearly inert at N ≈ 52. The per-layer/shared gap scales as O(N²·M·m); at N=52 versus the
  100–400-node probes where 5.5–6.8 s/it was measured, per-layer magnetic is affordable. **Do
  not import `G=4` from `bias_sharing` reflexively** — re-measure, and expect per-layer to be
  fine.
* *Optional, probably a distraction:* orienting bonds by a canonical rule (CIP rank) would
  reactivate the magnetic phase. Note and drop unless Tier A shows a spectral ceiling.

### 3.4 Everything else follows D3

`question_node: on` (§3.2.3 — `"isolated"` is rejected here, not aliased), `k_hop = 0`,
`prompt_style: chat` with instruct weights, LoRA dropout 0.15, and from §3.2.7 onward
`lr = 3e-5`, `bias_lr = 5e-3`, `lora_r = 16`. The question node carries the task instruction
**and the endpoint name** — that
is what makes Tox21's 12 endpoints and SIDER's 27 one model rather than 39 heads, and it is the
GTLM-natural choice: one example per `(molecule, endpoint)` pair, endpoint named in text.

RCM ordering on. Flex vs eager is genuinely open at N ≈ 52 — flex's win comes from block-skipping
and there is little to skip in a 52-node dense-ish graph, while molecules bucket into very few
`(L, N)` shapes so compile cost is low either way. Measure at M2; do not assume.

---

## 4. Multi-task design — the part that feeds the trunk

Three arms. All at 1B. All on machinery that **already exists**, which is why this is cheap:
`graphqa/load_dataset.py` merges `TextGraphDataset`s across tasks and carries a per-source
`ds_label`; `kgqa/config.py` has `train_datasets` / `eval_datasets` tuples with per-dataset
metric namespaces (`eval_{ds}_f1`) and a `selection_dataset` knob. Nothing new is needed.

| Arm | Training set | Question it answers |
|---|---|---|
| **1. Specialist** | one model per task | the per-task reference numbers; without these, arms 2–3 are uninterpretable |
| **2. Chemistry generalist** | all molecule tasks (A + B [+ C]) in one model, routed by the QUESTION node | within-domain interference and transfer — does `ring_membership` help BBBP? |
| **3. Cross-domain** | molecules folded into a Phase-1-style mixture (graphqa + probes + kgqa) | the actual admission-gate question: does chemistry cost the rest anything? |

Arm 2 is where the interesting result lives. The hypothesis worth pre-registering: **Tier A
transfers into Tier B.** Structural pretraining on free RDKit labels should improve scaffold-split
property prediction, because scaffold generalization *is* a structural-similarity problem. If it
does, that is a genuine result — a use for infinite free labels that the corpus-bound baselines
cannot copy — and it is directly measurable as arm 2 minus arm 1 on Tier B.

### 4.1 Held-out set — DECLARED 2026-08-28, before any run

This fills the slot `src/generalist/PLAN.md` §3.3 left open (*"Later: a scaffold-split molecule
set, one held-out CLRS algorithm family"*). Declared now, while no molecule result exists to bias
the choice — which is the only condition under which the declaration is worth anything.

**Held out from all molecule training, permanently:**

* **ClinTox** — one whole Tier-B dataset. Small (~1.5k, so cheap to forfeit) and structurally
  unlike the rest of Tier B, being a toxicity/trial-failure endpoint rather than a
  binding or permeability one.
* **`bond_path`** — one whole Tier-A family. Chosen for the same reason `direction` was chosen
  among the probes: it has a provable structural discriminator (SPD *is* the answer, by
  construction), so transfer there is unambiguous rather than a judgement call.

Both go into `registry.py` before the first mixture run. Neither appears in any training mixture,
in any arm, including the specialist arm — a specialist run on a held-out task would make the
arm-2-minus-arm-1 comparison in §4 meaningless for exactly the task it matters most on.

Scored the two ways trunk plan §3.3 requires: zero-shot transfer, and adaptation efficiency
(steps-to-target from the trunk vs from base Llama). The second is the one expected to carry
signal at 1B.

**Loss normalization (D7a).** Tier A answers are 1–3 tokens; Tier C captions are 50–100. Per-example
is the right default here, per the trunk plan's decided rule; if Tier C is included, it is the second
task in the repo (after CLRS-Text) that may want the per-task escape hatch. Record the choice.

---

## 5. Two experiments that do not depend on beating anyone

Both are cheap, both are eval-only or near-eval-only, and both produce claims that survive any
leaderboard outcome. Run them even if §2's gates fail.

**Size generalization.** Train on ZINC/MoleculeNet-scale molecules (≤ ~35 heavy atoms), test on
peptides / macrocycles (~150 atoms). This is the CLRS protocol from trunk plan §7 (*train at n ≤ 16,
test at n = 32/64*) applied to real chemistry, and it is a far stronger claim than in-distribution
accuracy. GTLM has a real shot here — the flat twin's SMILES string grows linearly and its
ring-closure digit bookkeeping degrades, while our bias is defined identically at any N.
Watch `max_spd` clamping (`project-khop-spd-shortcut`: the k=8 break was a `max_spd` artifact) —
at 150 atoms, paths exceed `max_spd=32`.

**Permutation invariance (§6).** See below. This one is nearly free.

---

## 6. The free win: atom-order invariance

GTLM is permutation-equivariant over prefix nodes by Property 1, verified to 2.77e-5
(`CLAUDE_CONTEXT.md` §2.3). **A SMILES-based LLM is not.** The same molecule written from a
different starting atom is a different token string — this is exactly why SMILES augmentation
exists as a standard trick in the cheminformatics literature.

The experiment: evaluate the flat twin on canonical SMILES and on 10 randomized SMILES per test
molecule; report the AUROC spread. GTLM's spread is provably zero (bounded by the Property 1
residual). Cost: eval only, one extra pass on the flat arm.

This is a **property claim, not a leaderboard claim**, which is why it is worth more than three
AUROC points. It is also the cleanest molecular statement of the paper's thesis: the graph arm
answers a question about a *molecule*, the flat arm answers a question about a *string that
happens to denote one*.

**One constraint on the effect size, found at M0.** Randomisation only produces variation where
the molecule is topologically asymmetric: benzene and cyclodecane have a single atom symmetry
class, so every traversal yields the same string and the flat arm is invariant *for free* on
them. So the measurement must be **restricted to, or at least stratified by, molecules with more
than one symmetry class** (`Chem.CanonicalRankAtoms(mol, breakTies=False)`), or symmetric
molecules will dilute the flat arm's spread toward zero and understate the effect. Pinned as a
test in `tests/experiments/molecules/test_encoding_roundtrip.py`.

---

## 7. Deferred, but do not let the schema preclude it: a 3D bias

The reason Uni-Mol and GEM win Tier B is 3D conformers. GTLM's bias is *any learned function of a
node pair* — so 3D is a drop-in new entry in `BIAS_TYPES`:

```
b_3D(u, v) = MLP( RBF( ‖x_u − x_v‖ ) )      # radial basis expansion of interatomic distance
```

Same shape as `SPDBias` (a distance → per-head scalar map), no new machinery, and it is
SE(3)-invariant by construction since only the distance enters. That would put GTLM on the same
information footing as the 3D specialists **while keeping the text channel** — which no 3D
specialist has.

**Not in cycle 1.** But two things must be true now so it stays possible: the schema must carry
optional per-node coordinates, and the dataset builder must retain conformers rather than
discarding them at parse time. Both are free today and expensive to retrofit.

Per `feedback-keep-core-gtlm-clean`, a 3D bias is a real `src/models/biases/` addition (like
`LINEAR_BIAS.md`), not something absorbed in an experiment package.

---

## 8. Build order

| | Milestone | Done when | Cost |
|---|---|---|---|
| **M0** | ✅ **done 2026-08-28.** `rdkit` installed `--no-deps` (self-contained wheel; nothing else moved, and the venv's pre-existing `torch_sparse`/CUDA-13 mismatch is untouched). `ogb` **not** needed — the nine MoleculeNet CSVs come straight from the DeepChem S3 bucket and the scaffold split is ours (`scaffold_split`), which avoids a heavy dependency that would have pulled on torch. Sizes, tokens and splits measured. | §3.1.1 | login node |
| **M1** | ✅ **done 2026-08-28.** `data.py`: RDKit `Mol` → networkx `DiGraph` with `text` + `prompt_node`, the three encoding cells, the flat SMILES serializer, the scaffold split, and `roundtrip_check`. 119 unit tests in `tests/experiments/molecules/`. | round-trip clean at each encoding's declared level, full corpus — §3.2.1 | CPU |
| **M2** | ✅ **done (jobs 134071 + 134111, 2026-08-28/29).** Tier A generator (`tasks.py`, 10 families), `dataset.py`, the experiment package, `000_smoke.jsonc` (6 runs) and the canary (4 runs; config since deleted — §3.2.3). Speed/memory settled (§3.1.2). The smoke's 0.000 graph accuracy was undertraining, not a defect — the unfixed baseline reaches 1.000 at 1250 steps (§3.2.3). Bias parameters verifiably leave their init (6.27 → 23.9). | s/it and peak GB ✅; flex-vs-eager ✅; Null-gate controls ✅; graph arm learns ✅. **New blocker for M4: `ring_membership` is saturated and cannot discriminate.** | ~2 GPU-h |
| **M2b** | ✅ **done 2026-08-29 (job 134202, 18 runs).** Tier-A difficulty screen: all nine runnable families × both arms under the canary's recipe. Added because M2 showed task choice was an untested assumption. Found the base-rate confound (§3.2.4.1) and forced `base_rate` / `eval_curve` / `still_improving` into every run record. | a shortlist meeting §3.2.4's criteria — **not delivered**: applied literally the criteria admit one family, because criterion 2 assumed the graph arm leads where there is headroom (§3.2.5) | ~9 GPU-h |
| **M2c** | ✅ **done 2026-08-29 (`004_extended`, the width analysis, `005_fgcount_ablation`, `006_recipe` — 32 runs).** The screen at 2500 steps (§3.2.5); `max_spd` measured and kept at 32 (§3.2.5.1); prompt edges verified directed (§3.2.5.2); the `fg_count` bias ablation (§3.2.6, since retracted); **and the recipe fix (§3.2.7)** — `longest_chain` flips to graph 0.989 vs flat 0.947. | the local-vs-global split explained, and the training recipe no longer a confound ✅ | ~30 GPU-h |
| **M2d** | ⬜ **next.** (a) `fg_count` × `bias=none` at the tuned recipe — the one run that retires §3.2.6's retracted null. (b) Re-screen the six non-saturated families at the tuned recipe, both arms, 5000 steps: §3.2.4's admission ranking was computed under a recipe that suppressed the graph arm by up to 13 points and cannot be reused as-is. | a shortlist M4 can trust, and a verdict on whether the bias channel is load-bearing here | ~14 GPU-h |
| **M3** | 🔄 **data path built 2026-08-28**, GPU runs pending M2. `tier_b.py` (scaffold split, one example per `(molecule, endpoint)`, endpoint named in the QUESTION node) + `evaluate.py` (the relbench margin readout ported: `logit(" Yes") − logit(" No")` in fp32, sigmoid before the threshold metrics, `n_distinct` / `tied_pair_fraction` in every record). Classification only — regression still needs `numeric_text`, and `validate()` refuses those sets rather than inventing a binary label. | BACE end-to-end, both arms, 3 seeds | ~10 GPU-h |
| **M4** | Encoding sweep: §3.2's **3 encoding cells** (`rich×levi`, `terse×levi`, `rich×atom_only`) × `±smiles` × 4 bias arms × 3 seeds, **on M2d's shortlist** + BACE/BBBP — never on a saturated family, and never at the stale recipe (§3.2.7). Run `terse×levi` first — it is the cheapest run in the sweep and it decides whether the featurizer is needed at all. | §3.2's arms decided and written into a frozen config | ~100 GPU-h *(estimate)* |
| **M5** | Multi-task arms 1 / 2 / 3 (§4). | arm 2 − arm 1 measured on Tier B | ~100 GPU-h *(estimate)* |
| **M6** | §5 and §6 experiments. | size-generalization curve + permutation spread | ~20 GPU-h |
| **M7** | Admission fork into the trunk, against §5's four criteria in the trunk plan. | pass/fail recorded in `lineage.json` | per trunk plan |

**M1's round-trip test is the highest-value test in the plan.** An encoding bug — a dropped
aromatic flag, a bond mapped to the wrong Levi node — is otherwise completely silent: the model
trains, the loss falls, the number is just mediocre, and six weeks later it looks like an
architectural limitation. The molecular equivalent of relbench's `test_evaluate_crosscheck.py`.

**Total ≈ 250 GPU-h** *(estimate)* — small next to Phase 1's 300–500. Molecules are the cheapest
domain in the mixture, so the "before the trunk" constraint is easy to satisfy; M0–M3 can run
alongside Phase 1 rather than blocking it. Per `feedback-submit-to-slurm`, nothing runs on the
login node except M0's download and stats.

---

## 9. Risks

* **Framing.** Reported as "GTLM on MoleculeNet", the honest outcome (lose to Uni-Mol, beat our
  flat twin) reads as failure. §0 and §2's gate table exist to fix the frame in advance. Same
  mistake relbench nearly made.
* **`stereocenters` and chirality more broadly.** Our graph genuinely holds less information
  than SMILES unless stereo is put in the node text. Decide explicitly and record it; a silent
  loss here would be misread as a structural-reasoning failure.
* **AUROC tie collapse on HIV.** ~3.5% positives plus bf16 margin quantization to 1/8
  (`project-gtlm-margin-quantization`) is the worst case for the margin readout. `n_distinct`
  and `tied_pair_fraction` in every run record from M3 onward, not retrofitted.
* **Scaffold-split variance.** Small sets (SIDER 1.4k, ClinTox 1.5k, FreeSolv 0.6k) swing by
  several AUROC points across seeds. ≥3 seeds, mean ± sd, and no per-dataset cherry-picking —
  the survey's own instability finding is the evidence for this.
* **Anchor incomparability.** InstructMol is 7B with a pretrained graph encoder; Uni-Mol is 3D +
  10M-molecule pretraining; Mol-LLM is a 23-task generalist on SELFIES. **Our matched control is
  our own flat twin**; everything in §2 is an external anchor. Say so in any writeup, exactly as
  relbench §1 caveat 1 does.
* **Levi changing the spectrum** (§3.2) — mitigated by the `atom_only` arm being in M4, not a
  follow-up.
* **Do not call a floor early** (`feedback-dont-call-floors-early`). Molecule sets are small and
  these runs will look flat before they move. Now twice-realised in this campaign: M2's 0.000 was
  undertraining (§3.2.3), and M2c's "plateau" at 0.855 gained 10 points at the same settings
  (§3.2.7). The `still_improving` flag installed after the first occurrence did not catch the
  second.
* **An inherited recipe is a confound until it is checked.** §3.2.7 spent ~30 GPU-h and four
  sweeps diagnosing an architectural deficit that was a copied hyperparameter block from the
  oldest experiment in the repo. The check costs one `grep` across `runs.jsonl` files and belongs
  **before** the first diagnostic sweep in any new domain, not after the fourth.
