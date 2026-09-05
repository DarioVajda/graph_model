# Molecules — CLOSED 2026-09-03

**Status: the specialist campaign is closed.** Tier A delivered a clean positive; Tier B delivered a
clean null. Nothing here is in flight. §10 holds the deferred plan for if and when this is picked up
again; §10.1's configs (`034`–`039`) are written and validated but unsubmitted, so picking it up is a
decision rather than a build.

**The one-paragraph verdict.** At a matched recipe over three datasets and nine paired seeds, the
graph arm and its SMILES flat twin are **indistinguishable — −0.0016, 95% CI [−0.023, +0.020]**
(§8.4.8). Gate B1 fails 0 of 3, exactly as §3.2.5 pre-registered before M3 ran. Both arms land in the
InstructMol-G band at 1/7 the parameters and beat every general-purpose-LLM row in §2.2, so gates B2
and B3 pass — but they pass for the *flat* arm too, so nothing in that comparison is attributable to
the structural channel. On **Tier A** the picture is the opposite and much stronger: 5 graph wins,
2 ties, 2 flat wins, split along *representational explicitness*, with the bias channel — not adapter
capacity — carrying the win (§3.2.8). Molecules were always the adverse domain for the headline
mechanism (§0) and the result is the one §0 predicted.

**This document was condensed on 2026-09-03** from a ~2400-line working log to a closed record. The
blow-by-blow — intermediate sweeps, superseded tables, day-by-day diagnosis — is in git history.
What survives here is: the framing, the anchors, the final numbers, the pre-registrations and how
they turned out, and the defects, because the defects are the most transferable thing the campaign
produced. Section numbers are preserved because other documents cite them.

---

## Configs

Results live in `results/<name>/runs.jsonl`. Sweeps `001`–`020` are the Tier-A campaign and the first
Tier-B measurement; `021`–`033` are the encoding verdict, the learning-rate correction and the
closing sweep. `034`–`039` are the deferred 3B/8B scale ladder (§10.1) — written, never submitted, so
they have no results directory.

| config | runs | what it settled | § |
|---|---:|---|---|
| `001_m3a_bace_bbbp` | 18 | first Tier-B measurement, `lr` 3e-5. **Superseded by the closing sweep** | 2.6 |
| `005`–`007` | 12 | recipe was the blocker; the bias is load-bearing and it is SPD | 3.2.7, 3.2.8 |
| `011_m2_loose_ends` | 6 | second seed on the ablation; ran gate A2's control, which exposed the split defect | 3.2.10 |
| `014_m2_rerun_molsplit` | 20 | **the Tier-A closing table** on molecule-disjoint splits | **3.2.5** |
| `015`+`019`+`020` | 8 | 3× budget changes nothing; B1's failure is not a budget artifact | 8.2 |
| `016_leakage_detector_restore` | 3 | **written, never run** — folds into the generalist as a validator | 9 |
| `021`–`025` | 36 | the encoding verdict at `lr` 3e-5 | 8.4.3 |
| `026`–`028` | 22 | `lr` 3e-4/1e-4 screen; supplies the closing sweep's graph seed 0 | 8.4.4, 8.4.5, 8.4.7 |
| `029`+`030` | 15 | closing sweep, **supplies the flat arm**; graph rows ran at `max_spd` 64 and are ablation | 8.4.8, 8.4.9 |
| `031_tier_b_closing_hiv_graph` | 3 | ⚠️ **1/3 — two cells OOM-killed at ~6 h.** Superseded; HIV has no clamp-ablation leg | 8.4.9 |
| `032_tier_b_closing` | 18 | 📄 **not run** — canonical config reproducing all eighteen closing cells | 8.4.6 |
| `033_tier_b_closing_graph_spd32` | 6 | the graph arm at `max_spd` 32; **completes the closing table at 18/18** | **8.4.8** |
| `034`–`039` | 88 | 📄 **written and validated, none submitted** — the 3B/8B scale ladder: prep, smoke, two `lr` screens, two closing sweeps | **10.1** |

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

> **Outcome against that target, 2026-09-03: 1 of 2.** The band was reached (§2.2, §8.4.8). The flat
> twin was not beaten. Point 3 is why the null is informative rather than embarrassing — a tie
> against a weak baseline says nothing, and this baseline beats the published 7B LLM row by 6–14
> points. Points 1, 2 and 4 are untouched by the Tier-B outcome and are what molecules still
> contribute to the trunk.

---

## 1. Benchmarks — three tiers

| tier | what | sets | metric | status |
|---|---|---|---|---|
| **A** | RDKit-generated structural questions — ten families, ground-truth labels, unlimited | generated over a MoleculeNet molecule pool | exact match | ✅ closed, §3.2.5 |
| **B** | MoleculeNet property prediction, scaffold split | **BACE, BBBP, HIV** (primary three) | ROC-AUC from the yes/no logit margin | ✅ closed, §8.4.8 |
| **C** | ChEBI-20 captioning, graph-to-SMILES | — | BLEU/ROUGE; validity + round-trip | deferred to the generalist |

**Deferred and why.** Tox21 and SIDER have no anchor ladder, so they are training signal rather than
results; they re-enter as generalist mixture sources. ESOL/FreeSolv/Lipophilicity are regression and
the margin readout cannot score a number. ClinTox and the `bond_path` family are **held out
permanently** (§4.1). QM9, peptides and text-to-molecule are out of scope.

---

## 2. Anchors, and what "beating them" means

### 2.2 The primary three — MoleculeNet classification, scaffold split, ROC-AUC ↑

**Every number below is as compiled in InstructMol's Table 2** (arXiv:2311.16208, current version,
read 2026-08-31), not re-run by us. A row's own paper may quote a different figure under a different
split. Four rows were run by InstructMol's own authors — **InstructMol-G, InstructMol-GS,
Llama-2-7B-chat and Vicuna-v1.3-7B**; every other row is copied from that method's own paper. Gates
B2 and B3 are both written against author-run rows, so **both rest on one paper's protocol and one
paper's execution** — a single point of failure, not four independent anchors.

InstructMol §4.1 uses DeepChem's `ScaffoldSplitter`, which is **deterministic** — `split()` accepts a
`seed` and never uses it. So their three seeds share one split and their ± is **training variance
only, the same quantity as ours**.

| | BACE | BBBP | HIV | what it is |
|---|---:|---:|---:|---|
| DMP (TF+GNN) | 89.4 | 77.8 | 81.4 | dual-view SMILES-transformer + GNN, consistency loss. The closest published thing to **our two arms fused** |
| **Uni-Mol** | **85.7** | **72.9** | **80.8** | SE(3) transformer over **3D conformers**, ~209M conformations. **The ceiling, not the target** — an information channel we structurally lack (§7) |
| MolFM | 83.9 | 72.9 | 78.8 | graph + text + KG, contrastively aligned |
| GraphMVP-C | 81.2 | 72.4 | 77.0 | 2D GNN, 3D used only in pretraining |
| MolCA (1D+2D) | 79.8 | 70.0 | — | Galactica + 2D GNN via Q-Former. **The graph-tokenizer architecture our thesis argues against** |
| KV-PLM | 78.5 | 70.5 | 71.8 | BERT over SMILES + biomedical text |
| MoMu | 76.7 | 70.5 | 75.9 | GNN + text encoder, contrastive |
| GraphCL | 75.3 | 69.7 | 78.5 | graph-only contrastive SSL, **no text channel** — the structure-only floor |
| ChemBERTa-2 | 73.5 | 69.8 | 79.3 | RoBERTa over 77M SMILES — the ceiling our flat twin approaches if Llama had been pretrained on SMILES |
| **InstructMol-G** (7B) | **84.3 ±0.6** | **68.6 ±0.3** | **74.0 ±0.1** | Vicuna-7B + frozen 2D graph encoder + projector, LoRA. **The band we should land in** — gate B2 |
| InstructMol-GS (7B) | 82.1 ±0.1 | 72.4 ±0.3 | 68.9 ±0.3 | same + SMILES in the prompt. **Our `+smiles` arm, measured by someone else:** +3.8 BBBP, −5.1 HIV |
| **Llama-2-7B-chat**, LoRA | **74.8** | **65.6** | **62.3** | general 7B LLM, SMILES-in-prompt, **no graph channel** — the nearest published analogue of our flat twin. Gate B3 |
| Vicuna-v1.3-7B, LoRA | 68.3 | 60.1 | 58.1 | **InstructMol's own backbone without the graph channel** — see §10.2 |
| Galactica-6.7B / 30B / 120B | 58.4 / 72.7 / 61.7 | 53.5 / 59.6 / 66.1 | 72.2 / 75.9 / 74.5 | prompted, no fine-tuning. **Scale without adaptation is not monotone** |
| Vicuna-v1.5-13b-16k, 4-shot | 49.2 | 52.7 | 50.5 | **the floor, and it is at chance** — these endpoints are not solvable by prompting |
| | | | | |
| **OURS — flat twin (1B)** | **82.2** | **71.6** | **76.2** | Llama-3.2-1B, SMILES, LoRA |
| **OURS — GTLM graph (1B)** | **82.0** | **70.6** | **76.9** | Llama-3.2-1B, `rich_levi` prefix nodes + SPD/magnetic bias, LoRA |

**Where we land: 6th–7th of seventeen on all three, consistently.** Below the 3D and multi-view
pretrained specialists, level with 2D-pretrained GNNs, above every general-purpose-LLM row. Ahead of
InstructMol-G on BBBP (+2.0) and HIV (+2.9) at 1/7 the parameters; behind on BACE (−2.3).

**Version drift, checked.** ar5iv serves an earlier InstructMol with -G at 85.9 / 64.0 / 74.0. The
numbers above are the current arXiv version. Do not "fix" this table against the stale mirror.

### 2.5 Pre-registered gates, and how they came out

| gate | statement | outcome |
|---|---|---|
| **A2** | Tier A: graph wins `stereo_potential`; on `stereo_assigned`, chance with `stereo_tags: off`, high with `on` | **First half PASSES** (`014`): 0.767 vs 0.724, **+0.043 (+2.2σ)**. **Second half UNMEASURABLE** on the `014` pool — the family's test split collapsed to a single answer (§3.2.10). The only measurement of it is `011`'s pre-fix 0.9928 → 0.8152, which is a large and correct off/on gap |
| **B1** | **Tier B: graph beats our own flat twin by ≥ 1 sd on ≥ 2 of the primary 3. This is the real result.** | **FAILS, 0 of 3 — final (§8.4.8).** BACE −0.0022, BBBP −0.0101, HIV +0.0075; pooled −0.0016, CI [−0.023, +0.020]. **§3.2.5 pre-registered this.** *Caveat on the gate itself:* it asks for a margin near 0.013–0.025 from a three-seed design whose per-dataset resolution is 0.077 — **B1 was never attainable as written** |
| **B2** | land in the InstructMol-G band | **PASSES**, both arms, at 1/7 the parameters |
| **B3** | beat Llama-2-7B-chat LoRA | **PASSES decisively**, both arms: flat +7.4 / +6.0 / +13.9 |

**B2 and B3 pass for the flat arm too, so neither is evidence about the graph channel.** They
establish that the flat twin is a credible baseline — which is what makes B1's null informative.

### 2.6 M3a — the first Tier-B measurement (`001`, 18/18, 2026-09-01)

> **Read at `lr` 3e-5 and nowhere else — superseded by §8.4.8.** Kept for the record of §3.2.5's
> pre-registered prediction and because it is the only place the `bias: none` control was measured
> on Tier B. Its seed-sd is rate-specific and must not be carried forward as a noise gate.

| set | our **flat** | our **graph** (`spd+magnetic`) | our **graph** (`bias: none`) |
|---|---:|---:|---:|
| BACE | **0.8357 ± 0.0207** | 0.8183 ± 0.0136 | 0.7833 ± 0.0013 |
| BBBP | **0.7048 ± 0.0168** | 0.6719 ± 0.0087 | 0.6929 ± 0.0118 |

The bias channel is worth **+0.035 (+3.6 sd)** over `bias: none` on BACE and **−0.021** on BBBP.
**That decomposition has never been re-measured at the tuned recipe** and is the cheapest outstanding
experiment in this document (§10.3).

---

## 3. Encoding — the decision this document exists to make

### 3.1 Default: `atom_levi_rich`

**Nodes = atoms. Node text = a short natural-language atom descriptor, never the bare symbol** —
`"carbon aromatic ring deg3 H1"` rather than `"C"`. **Bonds = Levi nodes**, text `"single bond"` etc.

Three reasons: **fairness** (it is exactly what a GNN gets in its atom feature vector, so a
bare-symbol encoding would handicap us against every baseline in §2); **cost** (bare symbols are the
worst corner of the nodes-per-token overhead curve); and **it is what makes this GTLM rather than a
graph transformer** — the backbone reads "aromatic", "carbonyl", "chiral" as language it knows.

Measured at M0 over all nine Tier-B datasets: BACE 34.1 atoms mean / 72.9 Levi nodes, 42 flat SMILES
tokens against 368 `rich_levi`. Nodes-per-token is **0.20 rich vs 0.61 terse — a 3× reduction, not
the 6× first claimed**, because Levi bond nodes carry short text in both arms. Cost of the whole
architecture on molecules is **3.3×** plain-LLM using each arm's best backend, 47 ms/example.
**Flex for the graph arm, eager for the controls** — with the bias on, flex is 1.46× faster and uses
20% less memory; with it off, eager wins.

*Two measurement gotchas worth keeping.* SIDER has a **492-atom** molecule, so `max_nodes` cannot be
set from the mean. And `atom_labels=True` sets an atom-map number on every atom, so the flat arm's
string on an atom-named family is `[O:1]1[CH2:2]...` — **3.8× the plain SMILES length**. Any cost
argument quoting "SMILES is 42 tokens" is quoting the wrong number for half the suite.

### 3.2 Encoding arms

Two axes: how much text a node carries (`rich` / `terse`), and whether a bond is a node at all
(`levi` / `atom_only`). **Three cells, not four** — `terse × atom_only` is information-destroying (a
double bond becomes indistinguishable from a single one) and is invalid rather than weak.

**Pre-registered prediction (2026-08-29, commit `eb63ba2`, before any encoding sweep ran):
`rich × levi` wins.** ✅ **Held** — see §8.4.3, §8.4.5 and the HIV caveat in §8.4.7.

**`+smiles` is an arm, never the default.** InstructMol-GS vs -G is +3.8 BBBP and −5.1 HIV. A SMILES
string in the prompt is a flat serialization sitting inside a graph arm, and the headline must be
`graph_only` or the claim is unclean in exactly the way the WebQSP triplet arm is
(`project-triplet-excluded-from-gtlm-claims`).

**Round-trip verified: 61,369 × 3 encodings, zero failures.** Four iterations to get there, each a
real finding — isotopes and radical electrons were missing from the rich node text; an explicit `[H]`
was double-counted (`RemoveAllHs`, not `RemoveHs`, which keeps stereo-defining H); and `terse` needed
its comparison level corrected to `labelled_graph`, because **aromaticity perception requires
hydrogen counts** and terse drops them, so 6.3% of drug-like molecules are not reconstructible as
chemical objects. A `terse` loss should be attributed to that before being read as evidence about
node-text richness.

### 3.2.5 THE TIER-A CLOSING TABLE (`014`, 20/20, 2026-09-01)

Ten families × both arms, one recipe (`lr 3e-5, lora_r 16, 5000 steps`), **molecule- and
scaffold-disjoint splits**. Exact match on 1000 test examples; σ is the two-proportion sampling bound.

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

**The line is representational explicitness, not topology.** `ring_size` +0.129 against `ring_count`
−0.077 — same rings, same molecules, opposite verdicts — so "topology" cannot separate them. What
does: a SMILES string marks every ring closure with an explicit paired digit, so *counting* rings is
a **lexical** operation on the string. Ring *size* is the number of atoms between a matched pair,
which requires traversing the string as a structure. Motifs fit the same rule — a functional group is
a short substring (`N(=O)=O` for nitro), which is why `fg_count` is the graph arm's worst family; and
`fg_atom_membership` is the motif question re-asked about a *named atom*, which breaks the lexical
shortcut and flips it to a graph win.

> **The rule: the flat arm wins what SMILES makes explicit as text; the graph arm wins what has to be
> computed from structure.** Falsifiable, and it predicts the `ring_size`/`ring_count` split rather
> than absorbing it. **Two standing predictions, neither measured:** the graph arm should win
> `bond_path` (held out under §4.1, so this is a prediction about a family we have deliberately never
> run), and a "how many bonds" family should go to the flat arm. Formed *after* seeing `ring_count`,
> so treat it as the leading hypothesis, not a settled result.

**Disclosures that belong beside this table wherever it is quoted.**

1. **One seed.** Every σ is a sampling bound at n=1000 and says nothing about seed-to-seed training
   variance, which Tier B measures at 0.012–0.037 ROC-AUC. The marginal cells — `stereo_potential`
   +2.2σ, `aromatic_ring` +1.4σ, `fg_presence` −0.8σ — are the ones a second seed could move. The
   four wins above +4σ are not.
2. **The arms are not equally tuned.** `bias_lr` is graph-arm-only by construction, so this is a
   tuned arm against an untuned one, and the flat arm keeps `lr 3e-5`. **Until a flat `lr` sweep is
   run, the claim is "beats our flat twin", never "beats a fully-tuned baseline."** (Tier B has since
   done exactly this and it mattered — §8.4.4.)
3. **Two families carry almost no signal.** `fg_count` and `fg_presence` sit at a 0.849 base rate,
   leaving 0.151 of headroom. `aromatic_ring` is at the ceiling in both arms.
4. **Convergence confirmed, not assumed.** All 20 runs peak between 0.00 and 0.79 of budget with none
   still improving.

### 3.2.7 The recipe was the blocker — the campaign's single biggest confound

Molecules inherited `lr=1e-5, bias_lr=1e-3, lora_r=8` from `expressiveness`, the oldest experiment in
the repo. Measured across every campaign's run records that is the **lowest-capacity recipe
anywhere** — 3–20× and 5–50× below graphqa, kgqa, relbench and probes. Moving to graphqa's
`lr=3e-5, bias_lr=5e-3, lora_r=16` — **applied identically to both arms** — flipped `longest_chain`
from −0.079 to **+0.042**, the first molecule-level graph win in the domain, with no change to
encoding, wiring or task. The change moved flat by −0.001 and graph by +0.030.

> **An inherited recipe is a confound until it is checked.** This cost ~30 GPU-h and four sweeps
> diagnosing an "architectural deficit" that was a copied hyperparameter block. The check is one
> `grep` across `runs.jsonl` files and belongs **before** the first diagnostic sweep in a new domain.
> The lesson recurred at Tier B eighteen sweeps later (§8.4.4).

### 3.2.8 The win is the bias, and the bias is SPD

§3.2.7 moved three knobs at once, admitting two readings: the bias finally reached useful magnitude,
or the adapter got enough capacity to read structure out of node text with the bias still inert. Four
runs settled it, and the two-seed novel-molecule version is:

| `longest_chain`, novel molecules | seed 0 | seed 1 |
|---|---:|---:|
| graph, `none` — **bias off, the control** | 0.7689 | 0.7576 |
| **flat twin** | 0.8030 | 0.8220 |
| graph, `spd` | 0.9356 | 0.9583 |
| **graph, `spd+magnetic` — the reported arm** | **0.9659** | **0.9697** |

**Adapter capacity alone does not reach parity with SMILES — it lands 3–6 points behind — and the
entire advantage over the flat twin is attributable to the structural channel.** The ordering
replicates at both seeds with every cell moving ≤0.023.

**It is SPD; magnetic adds nothing measurable** (`spd` alone reaches 0.983 of `spd+magnetic`'s
0.989). This **contradicts the plan's stated expectation** that magnetic should be load-bearing
because cycle detection is where spectral features classically win — an expectation that is now
tested and unsupported rather than untested. Molecules are undirected, so the magnetic
Laplacian's Hermitian phase is identically zero and `magnetic` degenerates to plain-Laplacian
spectral information — the `direction` probe's headline does **not** transfer here.

**The bias is a sample-efficiency win as much as an accuracy win:** `spd` reaches 0.950 val at step
800; `none` needs the full 5000. ~5.5× fewer steps.

**`fg_count`'s null is real and properly warranted:** `none` and `spd` are identical at 0.888 while
`bias_norm_final = 40.40` against `init = 0.0` — **the bias module trained to substantial magnitude
and still contributed nothing**, which is what `feedback-verify-nulls-are-real` demands.

*Caveat:* this ablation has **never been re-measured on a clean split at the tuned recipe.** It is
the decomposition behind the campaign's central claim (§10.3).

### 3.2.10 INCIDENT, CLOSED: Tier A's test split was not molecule-disjoint

**Found 2026-08-31, fixed same day, fully re-measured 2026-09-01 (`014`). The verdicts held and every
graph margin grew.** Kept as an incident record, not a live caveat.

**How it was found.** Gate A2's negative control ran for the first time and *fired*:
`stereo_assigned` with `stereo_tags: off` scored **0.936** against a 0.715 base rate where §1
predicts chance. No leak in the encoder — the cause was the data.

**The mechanism.** `generate_examples` drew from the molecule pool **with replacement**, and
`prepare_dataset` then sliced that list positionally into train/val/test. For a molecule-level family
the example is a deterministic function of the molecule, so a molecule recurring across the boundary
is an *exact* duplicate. **Up to 73.6% of test examples were exact duplicates of training items**,
and accuracy on the duplicate subset was 1.0000 in almost every run.

#### 3.2.10.1 The fix, and the pool it forced

**The fix:** split the molecules first by **Bemis–Murcko scaffold**, then generate examples inside
each split, consuming without replacement. Scaffold makes the test set *structurally* novel rather
than merely unseen, which is the property a structural-reasoning claim needs.
`SINGLE_EXAMPLE_TASKS` (`longest_chain`, `ring_count`, `stereo_potential`, `stereo_assigned`) now
**raise** rather than silently duplicate when asked for more examples than the split has molecules.
The artifact path carries a **`molsplit`** tag, so pre-fix `.gtds` files cannot be loaded by the
fixed code. Verified end to end at the sweeps' own sizes (4000/500/1000) across all ten families:
**zero** molecule overlap, **zero** scaffold overlap, **zero** train/test example overlap — against
the 73.6% duplicate rate the old sampler produced.
`tests/experiments/molecules/test_tier_a_splits.py` pins it, **including a test that reproduces the
old draw and requires the overlap to reappear** — a disjointness assert that cannot fail is
decoration.

**The pool it forced.** The old `bace,bbbp` pool (3552 molecules) became *infeasible* rather than
merely narrow — a proportional test pool is 646 molecules against 1000 requested — so `014` moved to
all five corpora (41232/5154/10308 molecules). That holds example counts fixed at 4000/500/1000 and
preserves the n=1000 test split, but **HIV is 41k of the 56k molecules**, so Tier A's chemistry is
now HIV-screening-dominated and absolute accuracies are not comparable with anything before it. The
graph − flat margin *is* comparable, because both arms draw the same molecules, questions and answers
in the same order — pinned by `test_both_arms_see_the_same_molecules_questions_and_answers`.

**Measured cost of that choice, which was larger than expected** — majority-class rate of the test
split, old pool → new: `stereo_assigned` 0.715 → **1.000 (VOID)**; `fg_count` and `fg_presence`
0.760 → 0.849 (headroom 0.240 → **0.151**); `longest_chain` 0.253 → 0.500; `stereo_potential`
0.285 → 0.527. The three ring families are unaffected. **In hindsight `bace,bbbp,tox21,lipo` was the
better pool and a future re-run should use it** — it is a balanced 15.5k, still far above the 1000
molecules a single-example family needs.

**Three lessons, all of which outlived the defect.**

1. **Post-hoc repair of a broken split is misleading in the direction it is supposed to be
   conservative about.** Re-scoring on the never-seen subset concluded contamination had *inflated*
   some margins; the clean re-run showed **every graph margin grew**. The contamination had been
   *understating* the graph arm, because the arm exploiting memorisable test items was the **flat**
   one. **Re-run instead.**
2. **Changing what a benchmark samples is a benchmark change.** The fix made the old pool infeasible,
   forcing a wider one, which pushed `stereo_assigned`'s test split to a **single answer** — the
   family scores 1.000 in both arms and is void. Two more families lost most of their headroom. Check
   base rates per family after any pool edit. *In hindsight `bace,bbbp,tox21,lipo` was the better
   pool and a future re-run should use it* — HIV at 73% of the five-corpus pool dominates every
   family's answer distribution.
3. **Untested data plumbing is where this class of defect lives.** Tier-A generation had **no test
   coverage at all**.

**Tier B is unaffected** — its scaffold split is molecule-disjoint by construction and by test.

### 3.4 Everything else follows D3

`question_node: on` (`"isolated"` is rejected here, not aliased), `k_hop = 0`, **LoRA dropout 0.05**,
RCM ordering on, `max_spd = 32` (§8.4.9), `lora_r = 16`, `bias_lr` 1e-2 graph / 5e-3 flat. Flex for
the graph arm, eager for the controls.

*Correction, 2026-09-04:* this line read "LoRA dropout 0.15" until now, which was never true here —
0.15 is the generalist's D3 value. No molecules config has ever set `lora_dropout`, so every run in
this document took `config.py`'s default of **0.05**, confirmed in the resolved configs under
`results/*/resolved/`. A re-run that wants to match §8.4.8 sets 0.05 explicitly.

**`node_position_mode` is unwired from this experiment entirely** — no config field, no flag, no run
record entry, with a test asserting it cannot be set. It has **two measurements and no positive
result**: KGQA E3 measured −9.4 F1 on WebQSP and the molecules canary put it 0.002 lower. A knob that
is wired but never set is worse than no knob — a sweep can put it in the axis list and produce a
campaign arm that was never justified.

**The base model is `meta-llama/Llama-3.2-1B` — raw pretrained weights, not the instruct variant**
(`config.py:22`, confirmed by the `model_name` on every run). Molecules has no `prompt_style` field.
Instruct weights are used only in KGQA. This matters for reading §2.2: InstructMol and the other LLM
baselines start from instruction-tuned checkpoints and we do not, which is a handicap we carry
deliberately — the structural bias is the object of study and an instruct base would confound what
the prefix nodes contribute.

**The learning rate is no longer 3e-5.** §8.4.6 settles it per (task, arm): **3e-4 on BACE and BBBP,
1e-4 on HIV, matched across arms on all three.** Tier-A numbers stand at 3e-5 as measured.

---

## 4. Multi-task design — the part that feeds the trunk

| arm | training set | question |
|---|---|---|
| **1. Specialist** | one model per task | the per-task reference numbers — ✅ this campaign |
| **2. Chemistry generalist** | all molecule tasks in one model, routed by the question node | **does Tier A transfer into Tier B?** |
| **3. Cross-domain** | molecules folded into the trunk mixture | the admission gate |

**Arm 2 is where the remaining interesting result lives**, and it is a *different* claim from B1:
structural pretraining on free RDKit labels should improve scaffold-split property prediction,
because scaffold generalization *is* a structural-similarity problem. It stays controlled because the
flat twin gets the same treatment, and it is directly measurable as arm 2 minus arm 1. Owned by
`src/generalist/MOLECULE_GENERALIST.md`.

### 4.1 Held-out set — DECLARED 2026-08-28, before any run

Declared while no molecule result existed to bias the choice — the only condition under which the
declaration is worth anything. **Held out from all molecule training, permanently, including the
specialist arm:**

* **ClinTox** — one whole Tier-B dataset. Small and structurally unlike the rest of Tier B.
* **`bond_path`** — one whole Tier-A family, chosen because it has a provable structural
  discriminator (SPD *is* the answer, by construction), so transfer there is unambiguous.

Both enforced in code (`dataset.py` refuses them), not remembered.

---

## 5. Two experiments that do not depend on beating anyone

Both cheap, both producing claims that survive any leaderboard outcome. **Neither has been run.**

**Size generalization.** Train on MoleculeNet-scale molecules (≤ ~35 heavy atoms), test on
peptides/macrocycles (~150 atoms). GTLM has a real shot: the flat twin's SMILES grows linearly and
its ring-closure bookkeeping degrades, while our bias is defined identically at any N. Watch
`max_spd` — §8.4.9 shows the clamp is inert at Tier-B sizes and says **nothing** at 150 atoms.

**Permutation invariance (§6).** Nearly free.

---

## 6. The free win: atom-order invariance

GTLM is permutation-equivariant over prefix nodes by Property 1, verified to 2.77e-5. **A
SMILES-based LLM is not** — the same molecule written from a different starting atom is a different
token string, which is exactly why SMILES augmentation exists as a standard trick.

The experiment: evaluate the flat twin on canonical SMILES and on 10 randomized SMILES per test
molecule; report the AUROC spread. GTLM's spread is provably zero. **Cost: one extra eval pass.**

This is a **property claim, not a leaderboard claim**, which is why it is worth more than three AUROC
points — and given §8.4.8's null it is now the strongest molecule-specific claim available. It is
also the cleanest statement of the thesis: the graph arm answers a question about a *molecule*, the
flat arm about a *string that happens to denote one*.

**One constraint on the effect size.** Randomisation only produces variation where the molecule is
topologically asymmetric — benzene has a single symmetry class, so every traversal yields the same
string and the flat arm is invariant for free. The measurement must be stratified by symmetry-class
count or symmetric molecules dilute the flat arm's spread toward zero. Pinned as a test.

---

## 7. Deferred, but do not let the schema preclude it: a 3D bias

The reason Uni-Mol and GEM win Tier B is 3D conformers. GTLM's bias is *any learned function of a
node pair*, so 3D is a drop-in entry in `BIAS_TYPES`:

```
b_3D(u, v) = MLP( RBF( ‖x_u − x_v‖ ) )      # radial basis expansion of interatomic distance
```

Same shape as `SPDBias`, no new machinery, SE(3)-invariant by construction. That would put GTLM on
the same information footing as the 3D specialists **while keeping the text channel**, which no 3D
specialist has. Two things must stay true for it to remain possible: the schema carries optional
per-node coordinates, and the dataset builder retains conformers rather than discarding them at parse
time. Both are free today and expensive to retrofit.

---

## 8. Results

| | milestone | outcome | cost |
|---|---|---|---|
| **M0–M1** | data, encodings, round-trip | ✅ 61,369 × 3 round trips, zero failures | CPU |
| **M2** | Tier-A generator + eleven sweeps | ✅ **5 graph wins / 2 ties / 2 flat / 1 void** (§3.2.5); the bias is load-bearing and it is SPD (§3.2.8) | ~75 GPU-h |
| **M3a/b** | Tier B, three datasets | ✅ **gate B1 fails 0/3** (§8.4.8) | ~40 GPU-h |
| **M4** | encoding sweep | ✅ `rich_levi` (§8.4.3, §8.4.5) | ~35 GPU-h |
| **M5** | multi-task arms 1/2/3 | ⬜ **the generalist** — `src/generalist/` | — |
| **M6** | §5 + §6 experiments | ⬜ **not run.** §6 is nearly free and is now the highest-value item here | ~20 GPU-h |

### 8.4 Checkpoint selection, and how validation behaves on Tier B

Three facts about the instrument, all measured, all of which bound every Tier-B number.

**1 — val and test are different populations, not two draws from one.** `scaffold_split` pours
scaffold groups into train → val → test sorted by `(len(group), group[0])` descending. Train takes
every multi-molecule group, so **val and test are 100% singleton scaffolds**, and among singletons
the tiebreak is **original CSV row order** — so val and test are disjoint contiguous slices of the
file. MoleculeNet's CSVs are ordered by source series, and the label is close to a step function of
row index on BACE and BBBP (positive rate by decile: BACE `0.67 0.65 1.00 1.00 1.00 0.25 0.00 0.00
0.00 0.00`). **HIV is not ordered** — the artifact is dataset-specific and spares the largest set.
This is DeepChem's split and is **not** changed, because §2.2's four author-run rows use it.

**2 — checkpoint selection is the casualty, and on BBBP validation ANTI-ranks the arms** (44%, i.e.
worse than chance). Selection is worth −0.017 on average. This is why **both `test_roc_auc` and
`test_roc_auc_last` are reported for every cell** in §8.4.8, and neither is promoted.

**3 — the test sets are too small to resolve the effects this campaign chases.** Hanley–McNeil s.e.
on test ROC-AUC: **BACE ±0.037** (152 molecules), **BBBP ±0.032** (204), **HIV ±0.024** (4112, but
only 132 positives). Every effect here is 0.02–0.05 — about one standard error. **No budget buys
precision; only a bigger test set does.**

*On protocol comparability:* at least four things are called "scaffold split". Ours is DeepChem's,
verified line by line. `pretrain-gnns`' differs only by `include_chirality=True`, which moves 8.1% of
BACE and 5.9% of BBBP between splits and — measured — **0.0% of HIV**, where the two lineages produce
the *same* split. Second-order against the s.e. above.

### 8.4.3 The encoding verdict, and §8.4.3.1's two claims

36 runs, three datasets, three seeds at `lr` 3e-5 found `rich_levi` and `rich_atom_only`
statistically tied (+0.0006 ± 0.0091), broken toward `rich_atom_only` on secondary criteria.
**That tie-break is RETIRED** — it held only at a learning rate below the useful range (§8.4.4). At
the tuned rate `rich_levi` wins 3 of 4 head-to-head cells on BACE/BBBP and beats `rich_atom_only` by
**0.11 on HIV** (§8.4.7). The §3.2 pre-registration is the one that survived.

#### 8.4.3.1 Two claims, and the difference between them

Conditioning matters and the two numbers must never be conflated:

* **On the benchmark as a fixed test set** — mean ± s.e. over seeds. **This is exactly the quantity
  §2.2's anchors publish** (InstructMol reports mean ± std over three seeds on one deterministic
  scaffold split), so a win of this kind is comparable to theirs on equal terms. It is what §8.4.8
  reports.
* **Generalising to new molecules** — the paired per-molecule bootstrap. Here **nothing in this
  campaign is resolved**: every per-cell 95% CI spans 0.

**A larger test set does not rescue this, and the expectation that it would was wrong.** AUROC
precision is set by the smaller class, so HIV's 4112 molecules buy nothing over BBBP's 204: HIV is
3.2% positive, giving a paired per-molecule CI of **±0.047 against BBBP's ±0.050**. Effective n is
~132 actives vs ~107.

Both numbers get reported. The bootstrap is a stricter bar than the field applies to itself and
should not be mistaken for our result being weaker than the anchors it is compared with.

### 8.4.4 Every Tier-B number to that point was measured at the wrong learning rate

Molecules ran Tier B at `lr` 3e-5, inherited via §3.2.7 from graphqa — the same class of confound,
recurring eighteen sweeps later. **Correct about absolute scores; misleading about the arm
comparison**, which is the only thing B1 turns on. Held at `rich_levi` so the rate is the only thing
moving, the arm gap barely responds (§8.4.8): +0.0039 BACE, +0.0070 BBBP, +0.0031 HIV. **The rate
correction raised both arms; it did not rescue ours.**

*Provenance warning:* this section's original BACE `lr` 1e-5/1e-4 rows came from an abandoned grid
whose records were **discarded** — no `runs.jsonl`, curve or per-example file. The direction is
corroborated by `026`; the specific rows are not reproducible from this repository and must not be
cited as if they were.

### 8.4.5 The screen — BACE + BBBP (`026`)

`lora_r` **32 is rejected: negative in 5 of 6 cells**, severe on flat (−0.0486 BACE, −0.0626 BBBP),
with the too-high-effective-lr signature. Doubling the rank doubles the parameter count taking 3e-4
steps, so r32 is partly a further lr increase rather than a clean capacity axis. **r16 is the setting;
the axis is closed.**

**The pre-registered selection rule fired against us and was reported as promised:** selecting on BACE
val picks flat r32 (test 0.8111) while the test-best cell is flat r16 (0.8598) — a **0.049** selection
cost. BACE val mis-ranks *within* an arm even though it ranks *across* arms at 83%.

> **The screen's headline BBBP win did not survive three seeds.** It was +0.0298 at one seed, and the
> flat arm's three-seed range (0.6983–0.7403) **contains** the 0.7306 that produced it. The section
> had flagged the cell as fragile *before* the seeds ran, on the grounds that the margin was smaller
> than its own selection gain (+0.0423) and the last-checkpoint comparison flipped sign — both
> correct. **A one-seed margin smaller than its own selection gain is not a finding.**

### 8.4.6 The closing sweep — rules fixed before the last screen cell landed

Written while `028` was still running, so none of it could be chosen after seeing the result it
applies to.

**Encoding `rich_levi`** by the §3.2 pre-registration. **`lora_r` 16.** **`bias_lr` 1e-2 graph /
5e-3 flat** — the one declared asymmetry, inert on the flat arm, which builds no bias channel.

**The learning rate is per (task, arm), and the rule was declared because it decides the headline:**

> Each `(task, arm)` gets the rate best supported for **that arm**. Where val and test disagree, the
> tie goes to whichever rate is better for the **flat** arm.

The second clause is selection on test applied **against our own interest**. It fires exactly once,
on BACE/flat, where val prefers 3e-5 (0.7497) and test prefers 3e-4 (0.8598) — taking val would have
converted a comfortable loss into a near-tie. **Every dataset ended up matched across arms:** 3e-4 on
BACE and BBBP, 1e-4 on HIV. That was not guaranteed by the per-arm rule; it is the result of breaking
both close calls toward the baseline.

**`max_spd` 32** — the value every sweep from `001` to `028` used. 64 was adopted by fiat for
`029`–`031` and **reversed**; see §8.4.9.

**Executed in four batches** — `029` (BACE+BBBP both arms), `030` (HIV flat), `031` (HIV graph at 64,
superseded), `033` (graph arm at 32, seeds 1–2). `032` is the canonical config reproducing all
eighteen cells. *The closing table is assembled from three sources, all at `max_spd` 32:* the flat arm
from `029`/`030` (inert there), the graph arm's seed 0 from `026`/`028`, seeds 1–2 from `033`.

### 8.4.7 The HIV screen — 3e-4 is wrong for HIV, and validation stops ranking the graph arm

| run | arm | encoding | `lr` | val | **test** | peak @ | still impr. |
|---|---|---|---:|---:|---:|---:|:---:|
| `027` | flat | — | 3e-4 | 0.7658 | 0.7447 | 8% | no |
| `027` | graph | `rich_atom_only` | 3e-4 | 0.7611 | 0.6927 | 90% | no |
| `027` | graph | `rich_levi` | 3e-4 | **0.7783** | **0.6754** | 100% | no |
| `028` | flat | — | 1e-4 | **0.8240** | 0.7485 | 34% | **yes** |
| `028` | graph | `rich_atom_only` | 1e-4 | 0.8207 | 0.6731 | 92% | no |
| `028` | graph | `rich_levi` | 1e-4 | 0.7860 | **0.7839** | 68% | **yes** |

**3e-4 costs the graph arm 0.109 on HIV** while the flat arm barely notices — the opposite of BACE
and BBBP, and why the rate is set per (task, arm). At 3e-4 the graph `rich_levi` cell has the
**highest validation score in the screen and the lowest test score**: validation is not merely noisy
there, it is pointed the wrong way.

> **The uncomfortable one: on HIV, validation does not rank the graph arm's encodings.** At 1e-4 val
> puts `rich_atom_only` **ahead** by 0.035; test puts `rich_levi` ahead by **0.111**. Selecting the
> encoding on validation — the rule used for every other hyperparameter — would have chosen
> `rich_atom_only` and closed HIV near 0.67. **The encoding choice therefore rests entirely on the
> §3.2 pre-registration** (commit `eb63ba2`, four days before these ran) and nothing else. A reader is
> entitled to note that the pre-registered choice happens to be the one with the better test number;
> the only answer is the commit date, which is why it is cited rather than asserted. **HIV does not
> corroborate the encoding verdict — it declines to rank the encodings in a way we may use.**

**The budget is binding at 1e-4:** both r16 cells finish `still_improving`. HIV's closing numbers are
floors for both arms, cut off at the same place.

### 8.4.8 THE CLOSING NUMBERS — all three datasets, 18/18 (2026-09-03)

Three seeds per cell at the settled recipe: `rich_levi`, `lora_r` 16, `max_spd` 32, `bias_lr` 1e-2
graph / 5e-3 flat, `data_seed` 0, `lr` matched across arms (3e-4 BACE/BBBP at 40 epochs; 1e-4 HIV at
10 epochs).

| set | arm | seed 0 | seed 1 | seed 2 | **mean** | sd | s.e. | last-ckpt |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BACE | flat | 0.8470 | 0.7975 | 0.8228 | **0.8224** | 0.0248 | 0.0143 | 0.8338 |
| BACE | graph | 0.8220 | 0.8313 | 0.8074 | **0.8202** | 0.0120 | 0.0069 | 0.8133 |
| BBBP | flat | 0.7084 | 0.7403 | 0.6983 | **0.7157** | 0.0219 | 0.0127 | 0.6870 |
| BBBP | graph | 0.7306 | 0.7007 | 0.6855 | **0.7056** | 0.0229 | 0.0132 | 0.6882 |
| HIV | flat | 0.7485 | 0.7746 | 0.7620 | **0.7617** | 0.0130 | 0.0075 | 0.7401 |
| HIV | graph | 0.7839 | 0.7503 | 0.7731 | **0.7691** | 0.0172 | 0.0099 | 0.7336 |

**Paired by seed, graph − flat** (positive = the graph arm wins):

| set | seed 0 | seed 1 | seed 2 | mean | 95% CI | t | wins | last-ckpt |
|---|---:|---:|---:|---:|---|---:|:---:|---:|
| BACE | −0.0250 | +0.0338 | −0.0154 | **−0.0022** | [−0.081, +0.076] | −0.12 | 1/3 | −0.0205 |
| BBBP | +0.0222 | −0.0396 | −0.0128 | **−0.0101** | [−0.087, +0.067] | −0.56 | 1/3 | +0.0012 |
| HIV | +0.0354 | −0.0242 | +0.0112 | **+0.0075** | [−0.067, +0.082] | +0.43 | 2/3 | −0.0065 |
| **pooled (9 seeds)** | | | | **−0.0016** | **[−0.023, +0.020]** | −0.17 (8 df) | 4/9 | |

**The verdict: the two arms are indistinguishable.** The point estimates disagree in sign across
datasets — flat ahead on BACE and BBBP, graph ahead on HIV — by margins smaller than the seed noise
in every case. **Gate B1 fails 0 of 3.**

**HIV is weaker than §8.4.3's version of it.** At 3e-5, `rich_atom_only` beat flat by +0.0159 with a
**consistent sign on all three seeds** — the campaign's only such win. At the tuned rate with
`rich_levi` the lead is +0.0075, the sign is not consistent, and the last checkpoint flips to
−0.0065. The direction survives the rate correction; the consistency, which was the interesting part,
does not.

**PER DATASET THE DESIGN CANNOT RESOLVE WHAT IT WAS ASKED TO RESOLVE.** The paired-difference sd is
0.0278, so three seeds detect a gap only above **0.077**:

| difference to detect | seeds per cell (80% power) |
|---:|---:|
| 0.05 | 3 |
| 0.03 | 7 |
| 0.02 | 16 |
| 0.01 | 61 |

Each dataset's "n.s." means *underpowered*, not *equal*. **Gate B1's "≥ 1 sd on ≥ 2 of 3" was never
attainable at three seeds** — a defect in the gate, never noticed because its resolution was never
computed against a measured seed-sd.

**Pooling the nine seeds is what makes the closing claim sayable.** The pooled CI is
**[−0.023, +0.020]**, resolution 0.021 — narrow enough to exclude any effect larger than about two
ROC-AUC points either way. That is the defensible closing statement: *across BACE, BBBP and HIV at a
matched recipe, the structural bias channel changes scaffold-split property prediction by less than
±0.02.* Pooling is legitimate because the three datasets share recipe, seeds, selection rule and
metric and the per-dataset effects are homogeneous; it is reported **alongside** the per-dataset rows,
never instead, because their point estimates disagree in sign.

**The `bias: none` control was not re-measured at the tuned recipe** (§2.6, §10.3). No claim about
the bias channel's contribution should be made from the closing numbers alone.

### 8.4.9 The `max_spd` clamp ablation — null, and single seeds said otherwise twice

| task | arm | mean Δtest (64 − 32) | t (2 df) | | seed-sd @32 | @64 |
|---|---|---:|---:|---|---:|---:|
| BACE | graph | −0.0379 | −1.34 | n.s. | 0.0120 | 0.0369 |
| BBBP | graph | −0.0016 | −0.09 | n.s. | 0.0229 | 0.0202 |
| HIV | graph | *not measured — n = 1* | | | | |

**Nothing is significant and BBBP's point estimate is zero.** The clamp is not the binding constraint
it was suspected of being — exactly what the arithmetic predicted, since the governing quantity is
the fraction of *pairs* past the ceiling (**2.56%** BACE, **1.20%** BBBP), not the 53% of *molecules*
an earlier reading quoted. The parameter cost of the change is 32 rows × 512 = **16,384 parameters**
against 11.4M trainable.

**The methodological finding is worth more than the result, because this comparison produced two
false signals in a row.** *First*, the seed-0-only comparison showed BACE −0.0430 and BBBP −0.0185
against a flat placebo floor of ±0.008–0.013, which looked like a large consistent effect; BACE's own
seed 2 later reversed the sign. *Second*, after two BACE seeds the 32 cells looked markedly more
stable (3.1× ratio) with a plausible mechanism attached — sparsely-trained far rows at 64 versus one
well-estimated far bucket at 32. **It did not replicate** (BBBP 0.9×), and the BACE gap is not
significant on its own terms (F = 9.5 against a 0.05 critical value of 19). Both signals were tempting
because both pointed the way the campaign wanted.

> **A comparison whose effect size is near the seed-sd is not readable at n = 1, and a mechanism
> invented to explain an n = 2 pattern is a story, not a finding.**

**Why the recipe is 32 even though the ablation is null.** Not because 32 measured better — it did
not, to any standard this document accepts. Because sweeps `001`–`028` all ran at 32, and moving the
closing sweep alone to 64 bought nothing while making the campaign's headline numbers the only ones
incomparable with everything else in the project. **A knob whose parameter cost is negligible is not
thereby free to change.**

**The placebo is what made the comparison readable.** `max_spd` sizes a table the flat arm never
builds, so its rows measure pure nondeterminism. HIV flat seed 0, run at both settings, returned
**0.7485067419 both times — bitwise identical.** BACE and BBBP flat moved −0.0128 and +0.0076 on the
same inert field, which is the H100 entering the allowed node set and nothing else. Any `max_spd`
reading smaller than that is unreadable.

**HIV has no ablation leg, and the reason is a memory leak in the evaluation loader.** `031` was
three cells and **two were OOM-killed** — seed 0 at 5h44 and seed 2 at 6h02, both at host RSS
134.16 GB against a 128G cgroup limit of 134.22 GB. Only seed 1 completed, at 134.15 GB — 0.05% of
headroom. All three ran at the ceiling and which one survived was luck. Cost: **11.8 GPU-h that ran
nearly to completion and produced nothing.**

For two months the reading was that HIV is simply a large corpus that needs a larger request, and
`032`/`033` were given 192G on that basis. **That reading was wrong, and 124.9 GB was never a
steady-state peak — it was a number still climbing.** `dataloader_persistent_workers` is not a
per-loader flag: it applies to the *evaluation* loader as well, and `Trainer.evaluate` calls
`accelerator.prepare` on every pass, which constructs a **new** `DataLoaderShard` each time
(`accelerate/data_loader.py:1290`). The previous shard is dropped without `_shutdown_workers()`, so
every evaluation forks a fresh persistent worker set and leaks the last one. Each leaked worker
retains ~0.41 GB private. Measured on one HIV graph cell run both ways:

| `num_workers` | cgroup peak | anon at peak | children across the run | anon trend |
|---:|---:|---:|---|---|
| 3 | 60.18 GB | 15.66 GB | 0→11→10→13→16→19→22→26→29 | climbs 3.19 → 15.66, never returns |
| 0 | 50.67 GB | 7.16 GB | returns to 1 after every eval | flat at ~6.5 GB over 700 steps |

Only the anon half is unreclaimable; the rest of both peaks is page cache the kernel would drop
under pressure. At ~1.05 GB of leaked anon per evaluation the mechanism accounts for the bulk of the
gap between the flat 50.7 GB baseline and `031`'s 124.9 GB — the extrapolation is loose, but nothing
else in the process is remotely that size. **The three standing suspicions were all small.** The
worker fan-out over the resident graph list is 1.56 GB of real cgroup charge at matched step; the
resident corpus is 2.35 GB (a 185 MB pickle at 12.7× expansion); and the flex/`torch.compile` shape
buffers — the leading candidate for most of those two months — cost ~4.6 GB once and are flat between
evaluations. `sacct` MaxRSS on FRIDA equals the cgroup peak rather than tree-RSS, so the 155 GB
tree-RSS figure that made the corpus look implicated was never what the killer read.

The fix is one line: `dataloader_persistent_workers=False` in `train.py`, unconditionally, with the
reason at the call site. HIV then fits inside 128G with room to spare, and 192G is a workaround for a
leak rather than a right-sized request.

*(The array-task index is not the seed: `136323_0` ran seed 1, `_1` ran seed 0, `_2` ran seed 2.
`sacct` reports array indices and the seed lives in the run label.)*

---

## 9. What to carry out of this campaign

**On quoting the numbers.**

* **Our matched control is our own flat twin**, never a published row. §2.2's four author-run rows
  come from **one paper**, so B2 and B3 share a single point of failure; the copied specialist rows
  mostly use a different scaffold-split convention.
* **Treat any single-set Tier-B difference under ~0.03 as unresolved** (§8.4). Prefer HIV (±0.024)
  for claims that have to carry weight, and prefer the pooled contrast over any single dataset.
* **Read Δ, not the level**, when comparing Tier-A numbers with anything before 2026-09-01.
* **Tier A does not predict Tier B.** Tier A labels are deterministic functions of the graph we hand
  the model; Tier B labels are noisy experimental measurements of properties partly not determined by
  the 2D graph at all.
* **Our own arms are not equally tuned on Tier A** — `bias_lr` is graph-only. On Tier B they are
  (§8.4.6), which is what makes §8.4.8 quotable.

**On instruments — the campaign's most transferable output.** Four defects, one shape:

| defect | what it did | § |
|---|---|---|
| Tier-A split not molecule-disjoint | up to **73.6%** of test examples were exact training duplicates | 3.2.10 |
| `base_rate` recorded as `null` | a score without its floor is uninterpretable; hidden behind a *louder* crash | 8.1 |
| `still_improving` inverted | reported overfitting as under-training; fired hardest on runs that had fallen furthest from their peak | 8.2 |
| host memory logged but never asserted on — a *rising* peak read as a large one | **11.8 GPU-h** of nearly-complete runs killed, then two months of raising `--mem` against a loader leak | 8.4.9 |

> **The common shape: a quantity that is only ever *read*, never *asserted on*, has no
> error-detecting surface.** The standing rule: every instrument gets a test that fails when the
> instrument is wrong, **including one that reproduces the defect** — a disjointness assert that
> cannot fail is decoration.

**Four more procedural lessons, each paid for.**

1. **An inherited recipe is a confound until checked** (§3.2.7, §8.4.4) — it cost ~30 GPU-h, then
   recurred eighteen sweeps later. One `grep` across `runs.jsonl` files, before the first diagnostic
   sweep in a new domain.
2. **Do not call a floor early** — realised twice: M2's 0.000 was undertraining, and a "plateau" at
   0.855 gained 10 points at the same settings. The flag installed after the first occurrence did not
   catch the second.
3. **Diff the emitted command, not the config**, whenever a re-run must match an existing sweep — a
   `.jsonc` cannot show defaults, and this caught a wrong `eval_steps` and a wrong `bias_lr` that
   would have biased results *toward* the conclusion then being tested.
4. **Smoke a new path before committing a sweep to it.** Twice paid for itself.

**The negative control is currently missing.** `stereo_assigned` is void on the `014` pool, so the
suite has **no working leakage detector** — the thing that caught §3.2.10 is not in it.
`016_leakage_detector_restore` is written against the measured restoring pool
(`bace,bbbp,tox21,lipo`, test base 0.732) and **pre-registers the pass line at 0.774**. It folds into
the generalist as a validator with the parity channel closed rather than running as a specialist
config, and that validator now exists: `leakage` in `src/generalist/evaluate/builtin.py`, in the
default set. It closes the channel at evaluation instead of training a second model, so it costs two
scoring passes rather than a run, and it computes the floor from the split it scores instead of
inheriting 0.732 — the generalist draws from the whole train-role partition, which is a different
pool. `016` itself stays unrun; if the specialist campaign is ever picked up again it is still the
stronger form of the measurement, because an off/on *training* contrast is not what an eval-time
ablation reproduces.

---

## 10. DEFERRED — the plan for if this is picked up again

**Not scheduled. Nothing below is committed, and no job has been submitted.** Recorded now, while the
numbers and their caveats are fresh, so that a future decision starts from measurements rather than
recollection. §10.1's configs (`034`–`039`) exist and validate; they are unsubmitted, and two of them
carry placeholder learning rates that a screen has to replace first.

The Tier-B null in §8.4.8 is a result *at 1B*. Two things could change it, and they are worth doing
together because they share the same port work.

### 10.1 Scale the base model — 3B and 8B

**Configs `034`–`039` are written and validated; none has been submitted.** Fifty-four ladder cells
plus the prep, the smoke and the two rate screens. Everything below is the design they encode, so a
future decision starts from a runnable object rather than a paragraph.

**The observation this rests on:** in a separate experiment the LLM-vs-GTLM comparison **changed sign
with backbone scale** — flat won at 1B, the graph arm won slightly at 4B, and the graph arm won
significantly at 12B. If that transfers, the 1B null here is a capacity threshold rather than a
verdict on the architecture, and every number in §8.4.8 is measuring the wrong point on the curve.

> ⚠️ **That observation is not recorded anywhere in this repository.** It exists as a recollection of
> a separate experiment, and it is the entire motivation for §10.1. Before it is cited in a write-up
> it needs a source or a re-measurement. The ladder is worth running either way — a measured curve
> replaces the recollection with data — but the *reason for running it* cannot be a number nobody can
> point at.

**Plan.** The closing sweep of §8.4.6 verbatim — three datasets, both arms, three seeds, matched `lr`
per (task, arm) — re-run on `Llama-3.2-3B` and `Llama-3.1-8B`, with `032` as the 1B leg unchanged so
all three legs come from one harness.

| config | runs | what it does |
|---|---:|---|
| `034_scale_prep` | 12 | build the Tier-B `.gtds` for both backbones (`model_name` is in the cache key) |
| `035_scale_smoke` | 6 | 60 steps per cell; the only outputs that matter are `peak_gb`, host RSS and s/step |
| `036_scale_screen_3b` | 17 | the `lr` bracket at 3B, plus one `bias_lr` cell on BACE |
| `037_scale_screen_8b` | 17 | the same at 8B |
| `038_scale_ladder_3b` | 18 | the closing sweep at 3B — **`lr` fields are placeholders until `036` lands** |
| `039_scale_ladder_8b` | 18 | the same at 8B, gated on `037` |
| `032_tier_b_closing` | 18 | the 1B leg, unchanged and already measured (§8.4.8) |

Order is `034 → 035 → {036, 037} → {038, 039}`. Two manual prerequisites: `hf download` both gated
repos before anything trains — otherwise the first of eighteen array tasks pulls 16 GB inside its own
allocation while seventeen siblings race it — and re-check whether `ixb6` still needs excluding.

**What must be re-tuned rather than inherited, and this is not optional.** §3.2.7 and §8.4.4 are the
same lesson twice: a recipe carried across a change in capacity is a confound. `lr`, `bias_lr` and
`lora_r` were settled *for 1B* and the optimum moves with model size — larger models generally want
lower `lr`. **Re-screen the rate at each scale, on both arms, and keep §8.4.6's rule that ties break
toward the flat arm.** Inheriting 3e-4 into 8B would reproduce this campaign's biggest mistake in a
new place.

**Where the brackets come from.** Two standard rules for how the optimal rate falls with size, and
they are close enough at 3B to agree and far enough apart at 8B to need bracketing: width-based μP
(`lr ∝ 1/d_model`, 2048 → 3072 → 4096) says ÷1.5 and ÷2.0, while `1/sqrt(params)` over
1.24B → 3.21B → 8.03B says ÷1.6 and ÷2.55. The screens centre on the latter rounded to ÷2 and ÷3 and
span an order of magnitude around it, so μP's prediction sits inside the grid rather than outside it:

| | BACE / BBBP | HIV |
|---|---|---|
| 1B (measured, §8.4.6) | 3e-4 | 1e-4 |
| 3B bracket | {5e-5, **1.5e-4**, 4.5e-4} | {**5e-5**, 1.5e-4} |
| 8B bracket | {3.3e-5, **1e-4**, 3e-4} | {**3.3e-5**, 1e-4} |

An argmax at the edge of a bracket means the bracket was wrong and the screen extends in that
direction before the ladder runs. **HIV keeps its own bracket** — at 1B, 3e-4 cost the graph arm
0.109 AUROC on HIV while the flat arm barely noticed (§8.4.7), the largest single effect in the
Tier-B campaign and a learning-rate effect. To keep the screens affordable HIV runs them at 3 of 10
epochs, which is a truncated proxy and is disclosed as one.

**`bias_lr` gets a bracket too, and the reason is specific rather than routine.** `head_dim` goes
64 → 128 between 1B and both larger backbones, so the scaled content logits shrink by √2 while a bias
of unchanged magnitude does not: the same bias is *relatively louder* at 3B and 8B. One extra graph
cell on BACE at each scale (`bias_lr` 3e-3 against 1e-2) says whether that matters. The bias's
parameter count barely moves — SPD is `max_spd × heads`, and heads go 32 → 24 → 32 — so this is a
question about logit scale, not about capacity.

**The epoch budget is deliberately NOT re-tuned.** 40 epochs on BACE/BBBP and 10 on HIV are 1B's
budgets and they stay, because the ladder compares *gaps across scales* and a per-scale budget puts a
second uncontrolled axis under that comparison. The cost is real and gets disclosed: HIV finished
`still_improving` on both arms at 1B (§8.4.7), so its numbers are floors cut off at the same place,
and a lower rate at 8B makes that more true. If the budget is ever raised it moves for both arms at
all three scales at once, including a 1B re-run.

**What this predicts, stated in advance so the result can be read honestly.**

* If the sign flips, the interesting quantity is **the gap as a function of scale**, not the win.
  Three points (1B, 3B, 8B) on a matched recipe is a curve; a single 8B win is an anecdote.
* **§0's void argument is orthogonal to scale and predicts the flip should NOT happen here.** The
  headline mechanism is preserving node text, and an atom is `C` — scaling the backbone does not
  create node text. So a flip on *molecules* would mean the structural channel pays off through some
  route other than text preservation, which is a more interesting finding than the flip itself and
  should be reported as such rather than folded into the thesis.
* If it does not flip, that is a real bound: the mechanism does not rescue itself with scale in a
  domain where nodes carry no text.

**Power, which this campaign got wrong and should not get wrong again.** Three seeds resolve only
gaps above 0.077 per dataset (§8.4.8). If the expected effect at 8B is ~0.02–0.03, the design needs
**7–16 seeds per cell**, not 3 — decide that *before* submitting, from the seed-sd measured at that
scale, not inherited from 1B.

**Cost.** The 1B column is measured — `train_runtime_s` from the runs that produced §8.4.8, median
over available seeds. The others are that column scaled.

| dataset | arm | 1B (measured) | 3B (×2.1) | 8B (×4.2) |
|---|---|---:|---:|---:|
| BACE, 1480 steps | flat | 0.39 h | 0.8 h | 1.6 h |
| BACE | graph | 0.77 h | 1.6 h | 3.2 h |
| BBBP, 2040 steps | flat | 0.50 h | 1.1 h | 2.1 h |
| BBBP | graph | 1.07 h | 2.2 h | 4.5 h |
| HIV, 10280 steps | flat | 3.1 h | 6.5 h | 13.0 h |
| HIV | graph | 6.2 h | 13.0 h | **26.0 h** |

| block | 1B | 3B | 8B | total |
|---|---:|---:|---:|---:|
| BACE, 6 runs | 3.5 | 7.3 | 14.6 | 25 |
| BBBP, 6 runs | 4.7 | 9.9 | 19.8 | 34 |
| HIV, 6 runs | 27.9 | 58.6 | 117.2 | **204** |
| ladder subtotal, 18 runs | **36** | **76** | **152** | **264 GPU-h** |
| rate screen | — (done) | ~30 | ~61 | ~91 |
| **total** | 36 | 106 | 213 | **~355 GPU-h** |

**HIV is 77% of the ladder**, so every cost decision here is really a decision about HIV. And the
seed count dominates everything: 7 seeds instead of 3 at 8B alone is **+203 GPU-h**. Run 3 seeds
across all three legs first, measure the paired sd *at each scale*, pool the nine as §8.4.8 does, and
extend seeds only where the pooled sign actually moves.

**Where the ×2.1 and ×4.2 come from, and how much to trust them.** Fitting `t ∝ params^0.76` to a
measured Gemma-3 GTLM ladder (GPU-s/step 4.86 → 14.14 → 30.14 over 1.0 → 3.88 → 11.77 B). Sublinear,
because fixed per-step costs dilute. Two corrections point opposite ways and roughly cancel: our
ladder's per-layer bias cost grows more slowly than that one's (layers × heads goes 512 → 672 → 1024
here against 104 → 272 → 768 there), while LoRA still backprops full depth so the backbone term is
undiminished. Treat the band as ×1.9–2.6 and ×3.6–5.0 until `035` measures it. Note this halves the
~265 GPU-h an earlier revision of this section predicted for 8B, which assumed roughly linear scaling.

**Memory and wall clock, both of which have already cost this campaign real time.** HIV graph peaked
at 124.9 GB host RAM and 57.8 GB GPU at 1B, against 43–75 GB and 21 GB for everything else. **The
host figure no longer applies**: it was the eval-loader worker leak, not the dataset, and with
`dataloader_persistent_workers=False` the same cell measures a 50.7 GB cgroup peak of which 7.2 GB is
unreclaimable, flat across the run (§8.4.9). Host memory is dataset-side and should not move up the
ladder in any case; GPU should, to ~75–90 GB at 3B and ~90–120 GB at 8B, which puts the 8B cells off
an 80 GB H100. That is a prediction and `035` exists to replace it
with a measurement before 264 GPU-h are committed — §8.4.9 is what happens otherwise. The 26 h HIV
cells at 8B also need `-t 36:00:00`; `032`'s 12 h limit would kill them at 46% complete. DDP is not
the fix — the sweep already has eighteen independent runs to spend cards on, and every rank
recompiles the same flex kernels unless the inductor cache is primed.

### 10.2 Port GTLM to Vicuna-v1.3-7B for the matched-backbone comparison

**The design, and why it is the right one.** InstructMol-G *is* Vicuna-v1.3-7B plus a graph
tokenizer, and §2.2 already carries Vicuna-v1.3-7B + LoRA on SMILES as its own author-run row. So
porting GTLM onto **their** backbone gives a three-way comparison on one base model where **two of
the three arms are published by their own authors**:

| Vicuna-v1.3-7B + … | BACE | BBBP | HIV | who ran it |
|---|---:|---:|---:|---|
| LoRA on SMILES | 68.3 | 60.1 | 58.1 | InstructMol's authors |
| **graph tokens** (InstructMol-G) | 84.3 | 68.6 | 74.0 | InstructMol's authors |
| **GTLM prefix nodes + structural bias** | ? | ? | ? | **the only arm we run** |

This is the tokenizer-vs-prefix comparison the thesis is actually about — structure compressed to
query vectors against structure entering as attention bias over uncompressed nodes — obtained without
us touching their code.

**Why NOT the obvious alternative of reimplementing their projector on our backbone.** Two reasons,
and the second is the stronger:

1. Reporting a number for someone else's architecture that *we* tuned is widely and correctly read as
   disingenuous, whichever way it comes out.
2. **The failure mode is asymmetric.** A bug or a mistuning in our reimplementation of their method
   looks exactly like a win for us. This campaign spent its entire length refusing to let our own arm
   win by an unfair margin (§8.4.6's tie-break rule, the `bias_lr` disclosure, §3.2.5 disclosure 2);
   the same discipline forbids publishing a weak number for a method we are not expert in. We control
   our arms; we do not control theirs.

**What has to be true for the comparison to be honest, and each is a real constraint.**

* **Match their protocol, not ours.** Their split, their epoch budget, their LoRA configuration where
  it is stated, their reported metric. Where a detail is unstated, say so and test the sensitivity
  rather than picking the flattering option.
* **Vicuna-v1.3 is instruction-tuned.** §3.4 records that molecules deliberately uses a *raw* base;
  this arm breaks that choice on purpose, and the resulting number is therefore **not** comparable
  with §8.4.8's. It answers a different question and belongs in its own table.
* **Report our flat twin on Vicuna-v1.3-7B as well.** Otherwise the arm inherits exactly the
  attribution problem §8.4.8 has — where a win could belong to the backbone rather than the
  mechanism. This makes it four arms, not three, and the fourth is cheap.
* Their row is one paper's execution (§2.2). If our Vicuna flat twin lands far from their 68.3, that
  discrepancy is the finding and must be reported before anything else is read.

**Cost estimate:** ~7× the 1B runtimes, so ~230 GPU-h for an 18-cell sweep plus the flat twin, before
rate screens. Prerequisite: GTLM's model wrapper has only been exercised on Llama and Gemma
backbones — the port itself is unscoped work and should be smoked on one cell before any sweep
(§9, lesson 4).

### 10.3 Two cheap things that should happen first regardless

Both are small, both close gaps that currently limit what §8.4.8 can claim, and neither depends on
any decision above.

1. **The `bias: none` control at the tuned recipe** — ~6 GPU-h. §2.6 measured the bias channel worth
   +0.035 on BACE and −0.021 on BBBP, but **only at `lr` 3e-5**, so the campaign cannot currently
   decompose its own graph arm into "structural channel" versus "extra adapter capacity" at the
   settled numbers. It is the one ablation that speaks directly to the thesis, and §3.2.8 — the
   Tier-A version of the same decomposition — has never been measured on a clean split either.
2. **The permutation-invariance measurement (§6)** — one extra eval pass. Given the Tier-B null this
   is now the strongest molecule-specific claim available, it is a *property* claim no amount of
   pretraining can beat, and it is very nearly free. Stratify by symmetry-class count or symmetric
   molecules dilute the flat arm's spread toward zero.
