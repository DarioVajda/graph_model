# The molecule specialist paper section — plan and execution log

**Opened 2026-09-17.** The specialist campaign closed on 2026-09-03 (`PLAN.md`) with three
Tier-B property sets measured and nothing generative. This file plans and carries out the work
that turns that into one concise paper section: four anchored rows, three seeds each, error
bars, compared against published numbers only.

The LaTeX section is written at the end of this file when every number in it exists.

---

## 1. Scope — what goes in the section, and what does not

Two tests decide a row: **is there a literature ladder to sit against**, and **can our readout
score it**. Four rows pass both.

| row | metric | status at open |
|---|---|---|
| BACE | ROC-AUC, scaffold split | ✅ measured, 3 seeds (`PLAN.md` §8.4.8) |
| BBBP | ROC-AUC, scaffold split | ✅ measured, 3 seeds |
| HIV | ROC-AUC, scaffold split | ✅ measured, 3 seeds |
| **ChEBI-20 captioning** | BLEU-2/4, ROUGE-L, METEOR | ❌ **no specialist exists** — this is the work |

**Excluded, each for a stated reason.**

* **Tox21, SIDER** — measured, but our headline `roc_auc` pools every `(molecule, endpoint)` row
  while the literature averages per-endpoint AUROCs. Recomputed under the published convention in
  §4 before any inclusion decision is taken; the anchors are 2D-GNN pretraining papers rather than
  LLM rows, which is a different ladder from the one the other three sets sit on.
* **ESOL, FreeSolv, Lipophilicity** — regression. The yes/no margin readout cannot score a number;
  they need the `numeric_text` path, which is not built. This is the honest gap in the suite and
  the section says so in half a line rather than pretending the suite is complete.
* **ClinTox** — held out permanently since 2026-08-28 (`data.py::HELD_OUT_DATASETS`). Never trained
  on, in any arm, including this one.
* **Tier-A structural probes, graph-to-SMILES** — no published ladder exists in this direction.
  They are the strongest molecule results we have and they belong to the mechanism section, not to
  a benchmark table whose whole premise is comparability.
* **Text-to-molecule** — GTLM emits text, not graphs. Out of scope by construction.

## 2. Backbones — one rung, and where the second would go

**BACE / BBBP / HIV stay at `Llama-3.2-1B`.** They are measured, and 1B is also the flattering
choice: the anchor table's general-purpose-LLM rows are all 7B, and we beat every one of them.
A 3B row would weaken "1/7 the parameters" and buy little, because scaffold-split property
prediction here is noise-bound rather than capacity-bound — `PLAN.md` §8.4.8 resolves nothing
under 0.077 per dataset at three seeds.

**ChEBI-20 runs on `Llama-3.2-1B-Instruct` in chat formatting.** Captioning is generation, and
generation needs the instruct paradigm (D3: instruct weights and chat template, both or neither).
`MOLECULE_GENERALIST.md` §7.4 measured captioning roughly doubling across that switch.

That leaves a seam — property rows on base weights, the caption row on instruct — and the section
discloses it in one clause rather than hiding it. Closing the seam means re-running three property
specialists on instruct weights with `lr` re-screened, ~24 GPU-h plus a screen, and it is not worth
it for a table whose property numbers are already the campaign's closing numbers.

**The 3B/8B ladder (`034`–`039`) stays unsubmitted.** Its entire motivation is a recollection
recorded nowhere (`PLAN.md` §10.1), and §0 predicts no sign flip on molecules. If scaling budget
ever appears it goes to ChEBI-20, the only row with headroom, and not to HIV, which is 77 % of the
ladder's cost.

## 3. The four things that had to be fixed before a ChEBI number could be published

Each was checked against the build manifests and the run records, not assumed.

1. **The test set was 2,889 of 3,300.** `chebi_heavy_atom_cap: 64` dropped 251 and
   `chebi_allow_disconnected: false` dropped 160. The dropped molecules are the large and the
   multi-fragment ones — the hard end of the split. Reporting BLEU on the easy 87.5 % against
   MolT5's 3,300 is not a comparison.
   **Fix:** the specialist builds at `chebi_heavy_atom_cap: 128`, `chebi_allow_disconnected: true`,
   and every uncovered molecule is charged against its real caption in the reported number. Coverage
   is disclosed beside it. **Verified against the benchmark file rather than predicted: 39 excluded
   of 3,300, coverage 0.9882, every one of them the heavy-atom cap and none of them disconnection.**
   That takes the coverage charge from 0.068 to roughly 0.006.
2. **The reported metrics ran on n = 500** — a subsample of that 2,889, in the trunk and in the
   anneal alike (`in_mixture` carries `max_samples: 500`).
   **Fix:** `molecules/chebi_score.py`, modelled on `generalist/tools/g2s_report.py`, generates the whole split
   and dumps predictions.
3. **METEOR was the exact-match stage only** (`DESIGN.md` §D7.3) — no stemmer, no synonyms — so it
   is a lower bound on the published definition and not comparable to a MolT5 number.
   **Fix:** `molecules/chebi_lit_metrics.py` rescores the dumped predictions under the published
   protocol with NLTK, offline, on CPU.
4. **No anchor table existed.** Compiled in §5 the way `PLAN.md` §2.2 was: one source, version
   checked, author-run rows marked.

## 4. Steps

Ordered so the GPU work starts first and the CPU work fills the wait.

| # | step | where | status |
|---|---|---|---|
| S1 | `chebi_specialist` mixture + validator presets | `config.py` | ✅ |
| S2 | `010_chebi_specialist.jsonc` — graph arm, 1B-Instruct, 3 seeds | `configs/probes/` | ✅ validates, 3 cells |
| S3 | `data_prep` at cap 128, disconnected allowed | sbatch, CPU | ▶ job 156447 |

**Build note, for whoever runs the next one.** `data_prep` was submitted CPU-only and the magnetic
spectral decomposition therefore ran on CPU — `compute_magnetic_*` takes `use_gpu=True` and falls
back on `torch.cuda.is_available()`, silently. At `chebi_heavy_atom_cap: 128` that is ~10 examples/s
over 23,346 train rows, so the build is ~45 minutes rather than the ~10 the cap-64 mixture took.
**Give the build a GPU** (`GPU=1`), with the caveat that the `.map` runs at `batch_size=1` — one
molecule per `eigh` — so the win is the device, not the batching, and is worth measuring rather than
assuming.
| S4 | budget from the built manifest, then the three seeds | chain.sh | ▶ 158198/158202/158206 |

**Resolved budget:** 208,435 examples over 5,000 steps at 41.7 examples/step — **8.93 epochs** over the
21,612-molecule train pool. That is ~1.5× the ChEBI exposure the generalist's 20 % share bought, at
five times the gradient share. Two things were checked rather than assumed before submitting: the
built `test` split is ChEBI-20's own (`_draw_chebi` filters by partition role on the *train* side
only, so the benchmark denominator does not drift and no test molecule reaches training), and
`validate`'s "mol/chebi20=9 passes" is benign — `generator_passes` materialises passes only for
generators, and a corpus re-walks the one built artifact.

**The campaign was scheduling-bound, not compute-bound, and the first submission got that wrong.**
Three cells at 2 cards and a 24-hour window sat behind four higher-priority jobs on a cluster with
every GPU on every node allocated and the soonest Blackwell release 4h46 away. Resubmitted at **one
card and four-hour windows over four chunks, all three started immediately** on backfill. A 24-hour
request is essentially unbackfillable and a two-card request needs both cards free on one node;
neither buys anything here, because the optimizer step is identical either way
(`accumulation_steps` 8→16 holds `micro_batch_tokens` at 1024) and §8.6 records that graph forks
were not faster on more ranks.

> **Two of the three cells were scheduled onto `ixb7`** — the node the first attempt had excluded as
> "sick". With `MELLANOX_VISIBLE_DEVICES=none` they start there cleanly. The exclusion would have
> cost two thirds of this campaign its immediate start, for a node that was never broken.

**The schedule is WSD plus a short anneal, not a single cosine run, and that is a departure from this
repository's own specialist convention.** Arm 1 — the BACE/BBBP/HIV specialists whose numbers this
section reports — trains through `molecules/train.py` on `cosine_with_min_lr` with a one-epoch linear
warmup and `min_lr = lr/10`, in one run. The ChEBI specialist instead gets 200 linear warmup steps, a
constant stable phase at `lr` 1e-4, and a 500-step cosine decay to `lr/10` delivered as an anneal
fork.

**The reason is availability, not preference: ChEBI-20 exists only in the generalist harness.** The
molecules package has no ChEBI task at all (`PLAN.md` §1 deferred Tier C), and the generalist harness
is WSD-native — its schedule is a list of absolute-step segments precisely because a *trunk* has no
horizon. That justification does not apply to a fixed 5,000-step single-task run, so the choice
deserves stating rather than inheriting.

Kept, for three reasons. WSD with a short anneal is designed to match cosine at matched budget, so
this is not a weaker schedule. It leaves mixture share as the *only* difference between this cell and
the generalist's ChEBI-20 numbers, which is the one internal comparison the row supports. And the
alternative is either a new, untested single-cosine path in `schedule.py` or porting ChEBI into the
molecules package — both larger risks than the shape of a decay curve. The segment design also
resumed cleanly across the interruption below, which a cosine curve defined against a total step
count would have handled less gracefully.

**A DISCLOSURE THAT TRAVELS WITH THESE THREE CELLS: the seeds were interrupted at different steps.**
The first submission was cancelled on the belief that it was still queued; it was not — those jobs
had been running for 2h33 (seeds 0 and 1) and 1h01 (seed 2). The cancellation therefore landed
mid-run, and each cell resumed from its last complete checkpoint: **seeds 0 and 1 from step 4,000 of
5,000, seed 2 from step 1,500**. `Schedule` appends a 200-step re-warm from 0.1 × LR at each resume,
so the three cells carry that re-warm at *different* points of the stable phase.

What it costs, stated exactly: the optimizer step is identical throughout (`tokens_per_step` 16,384
and `micro_batch_tokens` 1,024 on both sides of the interruption), the LR is otherwise constant
across WSD's stable phase, and the only asymmetry is where a 200-step ramp sits. It is the same
perturbation the four-chunk design introduces on purpose — a chunk boundary falls wherever four
hours lands, so no chained run in this repository has ever been free of it, and `rewarm_steps`
exists to absorb exactly this. It is disclosed rather than tuned away, and it is a reason to read
the three seeds' *spread* as slightly inflated rather than as pure seed noise.

The check that should have been made and was not: **re-read the queue immediately before cancelling**,
not from a snapshot taken when the jobs were submitted. `sacct -X --format=Start,Elapsed` answers it
in one call.
| S5 | anneal fork per seed (`forks/anneal_chebi_specialist.jsonc`) | run_cli.sh | ✅ written |
| S6 | `molecules/chebi_score.py` — whole split, predictions dumped | sbatch, 1 GPU | ✅ written |
| S7 | same on the six `008` generalist anneals (graph + flat, 3 seeds) | sbatch, 1 GPU | ▶ 156534, 156782–6 |

**The `ixb7` failures were a misdiagnosis, and the record should say so.** Ten scoring jobs died in
under 30 seconds with `enroot-mount: failed to mount: /dev/infiniband/rdma_cm`, and the first
reading was a sick node to be routed around with `--exclude`. It is not: `ixb7` lacks `rdma_ucm`,
and the container step only needs the enroot mellanox hook skipped —
`srun --export=ALL,MELLANOX_VISIBLE_DEVICES=none`, which the `tools/` launchers now pass, and which
holds for single-node jobs because they need no InfiniBand. Excluding the node would have cost a
whole B300 node's worth of scheduling for a one-line environment fix. Nothing here excludes a node.
| S8 | `molecules/chebi_lit_metrics.py` — published protocol, full-3300 accounting | CPU | ✅ written + tested |
| S9 | Tox21 / SIDER per-endpoint means; inclusion decision | CPU | ✅ excluded, §6 |
| S10 | the anchor table | — | ✅ §5 |
| S11 | property rows re-derived from `runs.jsonl` | CPU | ✅ §6 |
| S12 | assemble, then the LaTeX section at the end of this file | — | |

**Instruments added, each with a test that fails when it is wrong** (`PLAN.md` §9's standing rule):
`molecules/chebi_score.py` (whole-split generation, predictions dumped, coverage counted against the
benchmark's own 3,300), `molecules/chebi_lit_metrics.py` (the published metric protocol, offline) and
`tests/generalist/test_chebi_lit_metrics.py`, whose second case reproduces the defect — charging an
excluded molecule as a pair of empty strings instead of against its real caption — and fails if the
padding is reverted. 667 generalist tests pass with the change in.

## 5. Anchors — ChEBI-20 captioning

**Compiled 2026-09-17 from InstructMol's Table 3** (arXiv:2311.16208, HTML version), the same
source `PLAN.md` §2.2 takes the property anchors from, cross-checked against **MolCA's Table 2(b)**
(arXiv:2310.12798). Using one table for both halves of the section is deliberate: it is one
protocol and one execution, and §2.2 already discloses that as a single point of failure rather
than pretending four independent anchors.

Where the two sources overlap they agree to the digit — MolT5-Large 59.4 / 50.8 / 59.4 / 61.4 in
MolCA, MolT5-base 0.540 / 0.457 / 0.568 / 0.569 in InstructMol — which is the check that says the
two tables are the same benchmark scored the same way. MolCA is quoted on a 0–100 scale and
InstructMol on 0–1; everything below is 0–1.

| | BLEU-2 | BLEU-4 | ROUGE-L | METEOR | what it is |
|---|---:|---:|---:|---:|---|
| **Specialist models — SMILES or SMILES+graph in** | | | | | |
| MolT5-Base | 0.540 | 0.457 | 0.568 | 0.569 | T5-250M, full ft, SMILES→text. The reference implementation of the metric protocol |
| MolT5-Large | 0.594 | 0.508 | 0.594 | 0.614 | T5-780M, full ft |
| MoMu (MolT5-base) | 0.549 | 0.462 | — | 0.576 | + a GNN and a contrastive text encoder |
| MolFM (MolT5-base) | 0.585 | 0.498 | 0.594 | 0.607 | + graph, text and a KG |
| MolXPT | 0.594 | 0.505 | 0.597 | 0.626 | GPT pretrained on wrapped SMILES-text |
| Text+Chem T5-augm-base | 0.625 | 0.542 | 0.622 | 0.648 | multi-task chemistry T5 |
| **MolCA (Galac 1.3B, LoRA)** | **0.620** | **0.531** | **0.618** | **0.651** | Galactica-1.3B + 2D GNN via Q-Former. **The graph-tokenizer architecture our thesis argues against, at our own scale** |
| **Retrieval-augmented LLMs** | | | | | |
| GPT-3.5-turbo, 10-shot | 0.565 | 0.482 | 0.543 | 0.585 | MolReGPT-style retrieval, no fine-tuning |
| GPT-4-0314, 10-shot | 0.607 | 0.525 | 0.562 | 0.610 | same |
| **LLM generalists — the row our model is** | | | | | |
| Mol-Instruction (7B) | 0.249 | 0.171 | 0.289 | 0.271 | Llama-based, instruction-tuned on SMILES |
| BioMedGPT-10B | 0.234 | 0.141 | 0.332 | 0.308 | biomedical LLM |
| **InstructMol-G (7B)** | **0.466** | **0.365** | **0.479** | **0.491** | Vicuna-7B + frozen 2D graph encoder + projector, LoRA. **Graph in, no SMILES — the direct architectural comparator** |
| InstructMol-GS (7B) | 0.475 | 0.371 | 0.502 | 0.509 | same + SMILES in the prompt |

**What this table decides about the claim.** The specialist rows are chemistry-pretrained
sequence models reading SMILES, and Text+Chem T5 at 0.625 BLEU-2 is not a target a 1B Llama with
no chemical pretraining and no SMILES reaches. **InstructMol-G is the comparator that matters**:
same input condition (graph only), same adaptation (LoRA), 7× the parameters. Landing at or above
0.466 BLEU-2 from a 1B backbone is the reportable result; landing below it is also reportable, and
the section says which happened rather than choosing its opponent afterwards.

**Version drift, checked.** `PLAN.md` §2.2 records ar5iv serving a *stale* InstructMol whose
property numbers differ. The captioning table above is read from the current arXiv HTML and agrees
with MolCA's independently compiled overlap rows. Do not "fix" it against a mirror.

**One protocol caveat that is ours, not theirs.** Every row above is scored with NLTK `corpus_bleu`
and `meteor_score` over SciBERT word pieces and `rouge_score` over raw strings. Our own
`evaluate/captions.py` is none of those, so no number from it may be placed in this table;
`molecules/chebi_lit_metrics.py` exists to produce the row that may.

## 6. Results

### S10 — Tox21 and SIDER: the pooled number is not the published convention

Recomputed from the six `008` anneal records, which already carry the per-endpoint breakdown
beside the pooled figure. Mean ± sd over seeds 0/1/2:

| set | arm | pooled (ours) | mean over endpoints (published convention) | endpoints scored | unscorable |
|---|---|---:|---:|---:|---:|
| Tox21 | graph | 0.8373 ±0.0136 | **0.7685 ±0.0188** | 10 of 12 | 2 |
| Tox21 | flat | 0.8231 ±0.0078 | **0.7420 ±0.0266** | 10 of 12 | 2 |
| SIDER | graph | 0.8536 ±0.0063 | **0.6554 ±0.0095** | 22 of 27 | 5 |
| SIDER | flat | 0.8512 ±0.0055 | **0.7118 ±0.0085** | 24 of 27 | 3 |

**Pooling costs SIDER 0.20 of AUROC and Tox21 0.07.** Pooling every `(molecule, endpoint)` row into
one curve lets a model score by telling endpoints apart rather than by predicting any of them, and
the size of that effect tracks the endpoint count — 27 for SIDER, 12 for Tox21. Any comparison of
our Tier-B multi-endpoint numbers against a published row had to be made on the lower figure, and
never was.

**They stay out of the section, and now for a measured reason rather than an asserted one.** Three
things are wrong with them as a publishable row, and the third is fatal on its own:

1. The anchors are 2D-GNN pretraining papers, a different ladder from the LLM rows the other three
   sets sit against.
2. Both numbers come off a **500-row sample** — `in_mixture` carries `max_samples: 500`, so Tox21
   spreads ~40 rows over each of 12 endpoints. That is why 2 and 5 endpoints have a single-class
   test split and cannot be scored at all.
3. **The two arms do not score the same endpoint set** — 22 against 24 on SIDER — so the arm
   contrast is not like-for-like at any resolution.

`PLAN.md` §1 excluded them for want of an anchor ladder. That was right, and this is the number
that says so. Fixing 2 and 3 is a full-split re-eval, which is affordable; it is not done here
because 1 would still stand and because this section reports *specialists*, which Tox21 and SIDER
have never had.

### S7/S8 — what the three protocol defects were worth, measured

The generalist's graph arm, seed 0, annealed — the same checkpoint `MOLECULE_GENERALIST.md` §7.4
reports — scored four ways:

| what is being scored | metric | BLEU-2 | ROUGE-L | METEOR |
|---|---|---:|---:|---:|
| 500-row sample of the built split | ours | **0.4257** | 0.5050 | 0.4941 |
| all 2,889 built rows | ours | 0.4134 | 0.4955 | 0.4825 |
| all 2,889 built rows | published | 0.3992 | 0.4639 | 0.4453 |
| **all 3,300 benchmark rows** | **published** | **0.3310** | **0.4061** | **0.3899** |

**The number that could have gone into the paper was 0.0947 BLEU-2 too high**, and it decomposes:
the 500-row subsample was worth **+0.012**, our own metric definitions **+0.014**, and the missing
411 molecules **+0.068**. Coverage is by far the largest of the three and it is the one that was
invisible — the subsample and the metric were both at least written down somewhere.

**The coverage term is not a constant and does not shrink with a better model.** It arrives through
BLEU's brevity penalty and through the means in ROUGE-L and METEOR, so it scales with how much of
the split a build omits. That is the whole argument for the specialist building at
`chebi_heavy_atom_cap: 128` with disconnected molecules admitted: 39 missing instead of 411 turns a
0.068 charge into roughly 0.006.

Ladder over heavy-atom buckets, same cell: 0.4022 (0–20, n=1118), 0.4179 (21–40, n=1352), 0.4242
(41–64, n=419). **The arm does not degrade on larger molecules** — the excluded tail was never a
region of weakness, it was a region of silence. `empty` is 0.000 in every bucket, so the stop token
is behaving and no part of this is the §8.5 defect.

**The generalist's ChEBI-20 row, published protocol, three seeds per arm**, mean ± sd. `built` is
the 2,889 molecules this build admits; `benchmark` is all 3,300 with the rest charged as misses.

| arm | denominator | BLEU-2 | BLEU-4 | ROUGE-L | METEOR |
|---|---|---:|---:|---:|---:|
| graph | built (2,889) | 0.3970 ±0.0037 | 0.2865 ±0.0040 | 0.4614 ±0.0037 | 0.4430 ±0.0040 |
| graph | **benchmark (3,300)** | **0.3290 ±0.0028** | **0.2375 ±0.0032** | **0.4039 ±0.0032** | **0.3879 ±0.0035** |
| SMILES twin | built (2,889) | 0.4256 ±0.0039 | 0.3167 ±0.0032 | 0.4839 ±0.0024 | 0.4689 ±0.0030 |
| SMILES twin | **benchmark (3,300)** | **0.3533 ±0.0044** | **0.2628 ±0.0034** | **0.4236 ±0.0021** | **0.4105 ±0.0026** |

**The coverage charge is 0.068 on the graph arm and 0.072 on the flat one** — the same molecules are
missing from both, so it is a property of the build and not of the arm, and it does not change the
arm ordering.

**Three seeds are ample here, which is not true anywhere else in this campaign.** The graph arm's
seed sd is 0.0028 on BLEU-2 — an order of magnitude tighter, relative to the gap being discussed,
than the 0.0278 paired sd that made the Tier-B comparison underpowered (§8.4.8). A captioning
corpus of 3,300 scored generatively is simply a much better-resolved instrument than 150 scaffold-
split test molecules read through a logit margin. The flat arm keeps the lead it held in §7.4, in
the expected direction: a caption is text about a molecule, and the flat arm's molecule is already
text.

### S4/S6 — THE ChEBI-20 SPECIALIST, 3/3 seeds (2026-09-17)

Published protocol, whole 3,300-molecule benchmark split, 39 unencodable molecules charged against
their real captions. `built` is the 3,261 the cap-128 build admits.

| | BLEU-2 | BLEU-4 | ROUGE-L | METEOR |
|---|---:|---:|---:|---:|
| specialist, built (3,261) | 0.4539 ±0.0007 | 0.3511 ±0.0011 | 0.5048 ±0.0014 | 0.4956 ±0.0021 |
| **specialist, benchmark (3,300)** | **0.4430 ±0.0007** | **0.3427 ±0.0011** | **0.4988 ±0.0014** | **0.4898 ±0.0021** |
| generalist graph, benchmark | 0.3290 ±0.0028 | 0.2375 ±0.0032 | 0.4039 ±0.0032 | 0.3879 ±0.0035 |
| *InstructMol-G (7B, graph in)* | *0.466* | *0.365* | *0.479* | *0.491* |

Per seed on the benchmark denominator: 0.4422, 0.4436, 0.4431.

**The re-warm asymmetry did not register, and that is a measurement rather than a hope.** Seed 2
carries its 200-step re-warm at step 1,500 where seeds 0 and 1 carry theirs at 4,000, and it lands
between them on every one of the four metrics. The disclosure above stays — the schedules genuinely
differ and a reader is entitled to know — but the spread it might have inflated is ±0.0007 on
BLEU-2, so nothing in this row rests on it.

**Against the comparator that matters, a 1B graph-only model is level with a 7B one.** ROUGE-L
+0.019, METEOR −0.001, BLEU-2 −0.023, BLEU-4 −0.023 against InstructMol-G — same input condition
(graph in, no SMILES), same adaptation (LoRA), one seventh the parameters. Every other LLM row in §5
is 0.19–0.21 BLEU-2 behind. The chemistry-pretrained sequence specialists (Text+Chem T5 0.625,
MolCA 0.620, MolT5-Large 0.594) remain well ahead and are not claimed against.

**Specialising is worth +0.114 BLEU-2 over the generalist** (0.3290 → 0.4429), and three things
contribute: five times the gradient share, ~1.5× the ChEBI epochs, and the cap-128 build.

**They are not cleanly separable, and the tempting arithmetic is wrong.** The coverage *charge* —
what each model loses by being scored on 3,300 rather than on its own admitted set — is 0.011 here
against 0.068 for the generalist, and it is easy to read the 0.057 difference as "the build's
contribution". It is not: the two `built` numbers are computed over **different molecule sets**
(3,261 including the large tail against 2,889 excluding it), so they are not a common baseline to
difference from, and a cap-128 generalist would also have *trained* on those molecules rather than
merely being scored on them. What can be said without qualification is the end-to-end figure on one
fixed denominator — +0.114 on the benchmark's own 3,300 — and that roughly half of it is reach into
molecules the old build never attempted. Anything finer needs a cap-128 generalist, which nothing
requires.

The comparison is available at all only because the generalist checkpoints were **rescored through
this instrument** rather than quoted from §7.4; against §7.4's published figure the apparent gain
would have been +0.017 instead of +0.114, which is the same number of GPU-hours reaching the
opposite conclusion.

**The seed spread is 0.0010 on BLEU-2**, two orders of magnitude tighter than the 0.0278 paired sd
that made the Tier-B comparison underpowered. A 3,300-molecule generative split is simply a
better-resolved instrument than 150 scaffold-split molecules read through a logit margin, and it is
why three seeds are ample for this row and marginal for the property rows.

### S11 — the property rows, re-derived

`PLAN.md` §8.4.8's table is quoted in the section, so it was recomputed from the `runs.jsonl`
records rather than copied. Selecting on the closing recipe — `rich_levi`, `lora_r` 16, `max_spd`
32 (graph), `bias_lr` 1e-2 graph / 5e-3 flat, `lr` 3e-4 BACE/BBBP and 1e-4 HIV — reproduces all six
cells to four decimals, the three paired differences, and the pooled −0.0016 over nine seeds at
sd 0.0278.

| set | arm | mean | sd | s.e. | matches §8.4.8 |
|---|---|---:|---:|---:|---|
| BACE | flat | 0.8224 | 0.0248 | 0.0143 | ✅ |
| BACE | graph | 0.8202 | 0.0120 | 0.0069 | ✅ |
| BBBP | flat | 0.7157 | 0.0219 | 0.0127 | ✅ |
| BBBP | graph | 0.7056 | 0.0229 | 0.0132 | ✅ |
| HIV | flat | 0.7617 | 0.0130 | 0.0075 | ✅ |
| HIV | graph | 0.7691 | 0.0172 | 0.0099 | ✅ |

**One trap worth recording, because it nearly went into the paper.** A first pass that selected on
everything *except* `lora_r` picked up `026`'s `lora_r 32` screen cell for BBBP graph seed 0 and
returned 0.7150 where the closing table says 0.7306 — a 0.0052 shift in the reported mean and a
0.0082 shift in its sd, both in the direction that would have made the graph arm look worse. The
recipe is six fields and a query that pins five of them is not a query for the recipe.

## 6b. Tier C moves into this package (2026-09-17)

`PLAN.md` §1 listed ChEBI-20 as Tier C of this suite and deferred it to the generalist, because the
generalist was the only harness that could train a generative task. The specialist number therefore
lived in `src/generalist/`, beside a mixture and a cross-source partition it does not need, on a
backbone and a schedule no other specialist in this package uses. Tier C now trains here.

**What moved, and what it cost.** `load_chebi` and the caption metrics moved *down* into
`molecules/chebi.py` and `molecules/captions.py`; the generalist imports them back, which is the
direction the layering already ran (`generalist/adapters/molecules.py` has always imported
`experiments.molecules.data`). Both moves are behaviour-preserving and were checked rather than
assumed: the delegating `load_chebi` reproduces the cap-64 build's `test: 2889` with drops
`{heavy_atom_cap: 251, disconnected: 160}` and the cap-128 build's `test: 3261` with
`{heavy_atom_cap: 39}`, which are the manifests on disk.

*(One apparent discrepancy resolved: the oldest manifest, `42f7a14bed21f876`, reports 7 more train
molecules than the port keeps. It predates the `no_heavy_atoms` filter — the build the specialist
actually used, `b41d930e1cd608a6`, has `no_heavy_atoms: 7` and matches exactly.)*

**The new work is the generative path**, which this package had never needed: Tiers A and B are read
teacher-forced from one token's logits. Tier C adds multi-token answer supervision
(`make_caption_labels`), left-padded batched generation (`generate.py`), and dev selection plus test
reporting (`chebi_score.py`).

**Two instruments, because two things could go wrong silently.** `verify_caption_labels` decodes the
supervised span back and refuses the build unless it is exactly the caption — an off-by-one at the
answer boundary would train the model to predict the `A:` of its own prompt and drop the caption's
first word, and the only symptom would be a slightly worse loss. `_load_bias_weights` prints whether
it restored the structural-bias tensors, because PEFT's checkpoint carries LoRA only: loading the
adapter and stopping would score a model whose bias tables sit at initialisation, which is a silent
ablation of the exact channel the experiment is about.

### The partition was costing this benchmark 21 % of its training data

The generalist applies D3.3's cross-source partition, under which a ChEBI *training* molecule that a
MoleculeNet corpus claims as test is withheld. On the cap-128 build that is **2,184 molecules**: the
generalist specialist trained on 21,612 where ChEBI-20 offers 26,071.

That rule is correct for a model scored on BACE/BBBP/HIV *and* on captions. It is wrong for a
specialist reported only on ChEBI-20, and **no published baseline pays it** — MolT5 and everything
compiled in §5 train on the benchmark's own three files. Tier C here uses those files.

**So `041` is not a clean backbone control, and the section must not claim it is.** It moves two
things at once: instruct→base weights (with chat→`Q:/A:` formatting) and partitioned→own split
(+21 % data). The two push in opposite directions — base weights should hurt captioning, more data
should help — so a null result would be the least informative outcome rather than the most. What it
cleanly establishes is the *joint* effect of running Tier C the way this package runs every other
specialist, which is the question the port was asked to answer.

### The two runs

| | `041` control | `042` budget scaling |
|---|---|---|
| backbone | `Llama-3.2-1B`, `Q:/A:` | same |
| schedule | WSD + 10 % decay | warmup + cosine to `lr/10` |
| `lr` | 1e-4 | **2e-4** |
| budget | 9 epochs (~7.3k steps) | 12 epochs (~9.8k steps) |
| selection | last checkpoint | **best dev BLEU-2**, post hoc |
| seeds | 0/1/2 | 0/1/2 |

**The learning rate is raised because the two schedules spend their budget at different average
rates.** WSD runs 200 warmup, ~4,800 at full rate and a 500-step decay: mean 0.94 of peak. Warmup
plus cosine to `lr/10` averages 0.55. The ratio is **1.72**, so 2× holds the dose rather than the
peak. Holding the peak instead would have under-trained the cosine arm and called it a schedule
effect.

**Selection is on dev BLEU-2 and not on `eval_loss`.** A caption has no dev metric the Trainer can
compute, so the obvious move is to select on loss — and that is the defect `TIER_METRIC`'s docstring
already records once for this package: selecting on a different quantity than the one reported.
`042` keeps six checkpoints, restores none during training, and picks by generating on the val
split.

**The generation path is proven before it was needed.** Against a real cap-128 checkpoint at step 400
of 7,326: bias parameters restored, all 3,260 val rows loaded, captions generated in the right
register (BLEU-2 0.174, which is what a 5 %-trained model should look like). Running that smoke hours
early, rather than at the end of the campaign, is the whole reason the scoring step is not a risk.

**Eight defects found by smoking the path rather than by a six-cell sweep**, each surfacing within
about a minute of a job starting:

1. a staleness guard that did not know Tier C (it compared against Tier A's configured sizes);
2. a `test_roc_auc_last` lookup on a tier that has no such metric;
3. four config fields with no CLI flag — the sweep passes config keys as flags, so the first
   submission died in seconds on all three cells;
4. `num_warmup_steps` passed both in `lr_scheduler_kwargs` and by `Trainer` itself, which is a
   duplicate keyword rather than an override;
5. `sweep/execute.py` not passing `MELLANOX_VISIBLE_DEVICES=none`, so two cells died on `ixb7`'s
   missing `rdma_ucm` — fixed at the launcher, where it now protects every sweep in the repository,
   and the resubmitted cell runs on that same node;
6. bias weights looked for under the wrong filename: they are `bias_parameters.pt` and
   `models.io.load_bias_parameters` already knows how to restore them. Now **fatal** on a
   bias-carrying arm, because that loader returning `None` is otherwise indistinguishable from a
   successful no-op — and the failure mode is scoring a model whose bias tables sit at
   initialisation;
7. a job script written to the session's `/tmp` scratchpad, which compute nodes cannot see;
8. `_prompt_only` truncating the prompt node's `input_ids` without truncating `labels`. The
   collator's own length assertion caught it, which is the difference between a clear error and a
   batch whose generated tokens are silently misaligned with their positions.

Items 6 and 8 are the two that would have produced a *number* rather than a crash, and both were
caught by an instrument someone had already written for a different reason.

### THE TENTH, AND THE EXPENSIVE ONE: Tier C had no stop token

**§8.5 was reintroduced, by me, in this port.** `TextGraphDataset.tokenize` takes `add_eos` and
defaults it off — correct for Tiers A and B, whose answer is one teacher-forced token, and wrong for
a caption, which is generated. Tier C was ported without carrying the flag across, so the captions
were supervised as `'…icosatrienoic acid.'` with nothing after the full stop.

A model never shown an end-of-text token does not learn where a caption ends. It writes the right
thing and runs on to the generation cap, and every caption metric then scores **stopping** rather
than correctness. That is exactly what `generalist/MOLECULE_GENERALIST.md` §8.5 records: a graph arm
emitting the exactly-correct canonical SMILES 46.5 % of the time as a *prefix* of a runaway
generation, reported as `exact_match` 0.0000 for two months and read as "the arm cannot serialize a
graph".

**Cost: about 2.5 GPU-hours across six cells, killed mid-run.** Letting them finish would have been
worse — the numbers would have been publishable-looking and meaningless.

**How it surfaced, and why it nearly did not.** The dev-selection curve read BLEU-2 0.176 at step
1,400 and 0.194 at 2,800, against the instruct specialist's 0.443. There was a ready explanation
sitting right there — *base weights are worse at captioning, exactly as predicted before the run* —
and that explanation was the danger. The number agreed with the hypothesis, so it invited belief.
Checking the built tokens instead took one command and showed the prompt node ending in `13`, the
full stop, rather than `128001`.

> **A result that confirms what you predicted is the one to check hardest, not the one to accept.**

**The eight earlier defects were all caught by smoking, and this one was not**, because the smoke
established only that training *ran*. It never inspected what was being supervised. Two instruments
now close that gap, and both refuse rather than warn:

* `verify_generative_stop_token` — a generative tier whose answers do not end in the stop token
  fails the build. `tests/.../test_chebi_stop_token.py` reproduces the defect with the real token.
* the dataset path carries **`eos`**, so an artifact built before the fix cannot be silently reused —
  the same discipline `molsplit` carries for the Tier-A split defect.

**A ninth, found while the runs were in flight and deliberately not fixed under them: this package's
trainer has no rank-zero guard.** `042` runs at `gpus_per_config: 2`, so the body executes under
torchrun on two ranks and **both append a training record** — seed 2's cell wrote two rows to
`runs.jsonl`, differing only in a one-second runtime. The generalist harness added such a guard
explicitly (*"keep a fork's setup and its result on rank zero"*); this package never needed one
because no molecules sweep had run on more than one card before.

What it does and does not affect, checked rather than assumed: the checkpoints are clean (four of
them for seed 2, each carrying both `adapter_model.safetensors` and `bias_parameters.pt`), because
HF's `Trainer` already writes those from rank zero only. The damage is confined to `runs.jsonl`, plus
a doubled post-training evaluation. **Aggregation therefore de-duplicates on `sweep_run`**, and the
guard belongs in `train.py` the next time this package runs multi-GPU — patching it under six live
jobs would have been the more expensive mistake.

## 6c. THE ELEVENTH DEFECT: a "WSD" run that never annealed (2026-09-18)

The first `041`/`042` sweeps produced a clean-looking result — cosine ahead of WSD by +0.023 BLEU-2
on every metric, with half the seed spread — and it was wrong at the root. **`041` never ran its
decay.** Its final checkpoint sits at the 1e-4 peak:

```
041 (labelled WSD)   step 1 → 1.012e-05    step 3663 (final) → 1.0e-04   ← peak, flat
042 (cosine)         step 1 → 2.457e-07    step 4884 (final) → 2.0e-05   ← lr/10, correct
```

`train.py`'s `steps_per_epoch` divided by `batch_size` and `accumulation_steps` but **not by
`world_size`**. An optimizer step consumes the product of all three, so at two ranks the Trainer runs
half the predicted steps. The WSD segments were laid out against 26,071 // 4 // 8 × 9 = **7,326**
steps — warmup 732, stable 5,861, decay 733 — while the run executed **3,663**. The decay segment
began at step 6,593, which is 2,930 steps after the run was over. The warmup arithmetic confirms it
exactly: `1e-5 + 9e-5 × (1/732)` = 1.0123e-05, matching the logged step-1 LR to every digit.

**Why it survived inspection.** Nothing about the run announces it. The loss curve is ordinary, the
checkpoint is real, the config says `"lr_schedule": "wsd"`, and the *run length* is correct — 3,663
steps either way, because `num_epochs` governs that and only the schedule laid over it moved. It was
found by accident, while checking the batch arithmetic for a four-card cell.

**What it cost.** Both headline conclusions. A schedule comparison between an annealed arm and a
never-annealed one measures mostly the missing anneal, which is the single best-understood
end-of-training gain there is — so "+0.023 for cosine" was not a schedule result. And `probes/010`
*did* anneal, via an explicit fork, so the 0.043 instruct-over-base gap was inflated by the same
missing anneal and was not a backbone result either.

Fixed in `train.py::steps_per_epoch_for`, with the stranding reproduced in
`tests/experiments/molecules/test_schedule_steps.py` (7 tests). **Containment is total by luck, not
design**: every other config in this package is `gpus_per_config: 1`, where the divisor is 1 and the
bug cannot fire, so the two Tier-C sweeps are the only runs it ever reached. `042` was touched far
more gently — cosine derives its decay from the Trainer's own step count, so the curve was right and
only `warmup_steps` was wrong (814, two real epochs, where the recipe says one).

Pre-fix results are kept under `results/_superseded_noanneal/` and **must not be set beside the
corrected ones**. For the record, they were: `041` 0.3999 ±0.0030, `042` 0.4229 ±0.0015.

## 6d. The 2×2 — backbone × schedule, all four cells on corrected code (2026-09-18)

The fix makes the pre-fix cells incomparable to any new one, so all four are run together rather than
patched piecemeal. This also finally closes the corner that was never run: **instruct weights and the
cosine recipe at the same time.**

| | WSD, 9 ep, `lr` 1e-4 | cosine, 12 ep, `lr` 2e-4 |
|---|---|---|
| **1B base**, `Q:/A:` | `041` | `042` |
| **1B Instruct**, chat | `043` | `044` |

Every cell: ChEBI's own three-file split, cap-128 build, `rich_levi` graph arm, LoRA r16, 3 seeds,
2 GPUs. The WSD cells are read at the end of the schedule; the cosine cells at their best **dev**
BLEU-2 checkpoint. `043`/`044` move only `model_name` from their base twin — `prompt_style` resolves
from the name, so instruct weights and the chat template arrive together or not at all.

**Why `043` is worth three cells on its own.** Without it the instruct claim still rests on
`probes/010`, which differs from the base runs in four ways at once — backbone, turn spelling, split
and harness — and two of those push in opposite directions, since `010` trained on the D3.3
partition's 21,612 molecules where this package trains on ChEBI's own 26,071. `044` alone would give
a headline number but could not separate "instruct helps" from "instruct helps *under cosine*".

**The budget stays at 12 epochs.** The pre-fix `042` dev curves had already flattened — two of three
seeds peaked at step 4,200 of 4,884 and then *declined*, and 2,800 → 4,200 bought +0.003 / +0.012 /
+0.013 BLEU-2. An earlier draft of this file read that as "still not converged" on the strength of
the one seed that picked the last checkpoint; the other two contradict it. More budget is not where
the remaining headroom is.

### The stop token differs between the two backbones, and both are consistent

Checked before the instruct cells could report anything, because §8.5 is this project's most expensive
recurring defect and the chat path had never been exercised in this package:

| backbone | trained stop token (`tokenizer.eos_token_id`) | `generate` halts on |
|---|---|---|
| `Llama-3.2-1B` | 128001 `<\|end_of_text\|>` | 128001 |
| `Llama-3.2-1B-Instruct` | **128009** `<\|eot_id\|>` | [128001, 128008, **128009**] |

`tokenize(add_eos=True)` appends the tokenizer's own EOS, so the two builds supervise *different*
tokens — and `generate` falls back to the model's `generation_config`, whose instruct entry is a
three-token set that contains 128009. The two agree on both backbones. Had the instruct build been
supervised with 128001 instead, every caption would have run past its end and the failure would have
looked like a quality problem rather than a stopping one.

### What this design does and does not license

Worth stating before the numbers land, so the reading is fixed in advance rather than fitted to the
result.

**Clean — the backbone axis.** `043` vs `041` and `044` vs `042` each move `model_name` and nothing
else; `prompt_style` resolves from the name, so instruct weights and the chat template arrive
together. Verified at the formatter: the base backbone gets `answer_prefix` `"\nA:"`, the instruct
backbone gets `"<|start_header_id|>assistant<|end_header_id|>\n\n"`. This is a real main effect, and
it is the one `probes/010` could never support.

**NOT clean — the schedule axis is a RECIPE axis.** The cosine cells differ from the WSD cells in
three ways at once: schedule, budget (12 epochs against 9), and learning rate (2e-4 against 1e-4,
`bias_lr` likewise). So the contrast supports "this package's recipe beats the generalist harness's
recipe" and **not** "cosine beats WSD". Turning it into a schedule ablation needs a matched-budget
cell — WSD at 12 epochs — which nothing in the paper needs, and which the pre-fix dev curves argue
against: 2,800 -> 4,884 steps bought +0.003 / +0.012 / +0.013 BLEU-2, so the budget is not what is
doing the work.

**Disclosed asymmetry — only the cosine arm selects on dev.** WSD's last checkpoint *is* its
schedule's answer; a cosine run's is not privileged, which is why the two are read differently. But
that hands the cosine arm a free selection advantage, and it points the same way as the conclusion it
supports, so it is measured rather than waved at. On the pre-fix dev curves it is worth **+0.0034
BLEU-2** (per seed +0.0064 / +0.0040 / 0.0000). Each cosine cell's LAST checkpoint is therefore also
scored on test, so the recipe comparison can be read with the selection removed as well as with it.

**The 2x LR correction survives the anneal fix.** Integrated over the schedules as they actually run,
the fixed WSD spends a mean 0.9101 of peak and warmup-plus-cosine 0.5458 — ratio **1.667**, against
the 1.72 computed when WSD was stuck at constant peak. The defect moved the justification by 0.05, so
2x stands as a deliberate slight over-correction.

### THE TWELFTH DEFECT: the instruct cells were discarding their own captions

Found while reading `043`'s first three seeds, which came in at 0.3434 ±0.0584 — a spread thirty
times the base control's over eval_loss curves that agreed to three decimal places (0.5962 / 0.5938 /
0.5896). A seed spread that large over training that identical is a symptom, not a result.

Greedy decoding was free to choose the stop token at the **first generated position**. That decodes,
under `skip_special_tokens=True`, to the empty string, and is scored as a total miss. It is nearly
invisible upstream: one position out of ~50, so even a 24 % probability there moves `eval_loss` by
about 0.005.

| | empty captions of 3,261 |
|---|---|
| `041`/`042`, base backbone, all six cells | **0** |
| `probes/010`, instruct, all three seeds | **0** |
| `043`, instruct, seeds 0/1/2 | **375 / 189 / 788** |

The asymmetry is the chat template: `<|start_header_id|>assistant<|end_header_id|>\n\n` followed
straight by `<|eot_id|>` is the instruct-tuned spelling of an empty turn, and the base backbone
carries no such prior. On a 200-row sample of the worst seed, forbidding the stop token for five
steps took empties from 46 to 0 and BLEU-2 from 0.3003 to 0.3914, and the recovered rows read as
ordinary captions.

`DEFAULT_MIN_NEW_TOKENS = 5` now applies on **every** arm. A constraint switched on for the arm it
rescues is not a measurement, and it is verifiably free where it is not needed: `041` seed2 rescored
to 0.4255, identical to its pre-fix value. `tests/experiments/molecules/test_generate_min_new.py`
reproduces the defect and asserts that generation never branches on the backbone.

**Why this one was dangerous.** It produced exactly the result I had already half-written — "the
instruct backbone is worse" — and I reported that reading, along with an inference about `010`'s
budget, before checking it. Both had to be withdrawn. The number agreed with the hypothesis, which is
the condition under which a number gets the least scrutiny. This is the same trap as §8.5 and as the
stop-token port, three times now in one campaign.

All twelve cells were regenerated afterwards into `results/chebi_v2/` as one homogeneous batch —
including redoing the cosine arms' dev selection, since those winners were chosen by generating too.
Nothing from the mixed directory is reused; the pre-fix scores are in
`results/_superseded_noeosguard/`.

### RESULTS — the 2x2, all twelve cells on corrected code (2026-09-18)

Published protocol, whole 3,300-molecule benchmark split, 39 unencodable molecules charged against
their real captions. Three seeds each. WSD cells are read at the end of the schedule; cosine cells at
their best dev BLEU-2 checkpoint.

| | WSD, 9 ep, `lr` 1e-4 | cosine, 12 ep, `lr` 2e-4 |
|---|---|---|
| **1B base**, `Q:/A:` | 0.4079 ±0.0020 | **0.4226 ±0.0101** |
| **1B Instruct**, chat | 0.3788 ±0.0214 | 0.3613 ±0.0217 |

Full metrics:

| run | BLEU-2 | BLEU-4 | ROUGE-L | METEOR |
|---|---:|---:|---:|---:|
| `041` base, WSD | 0.4079 ±0.0020 | 0.2994 | 0.4419 | 0.4492 |
| `042` base, cosine | **0.4226 ±0.0101** | **0.3180** | **0.4557** | **0.4596** |
| `043` instruct, WSD | 0.3788 ±0.0214 | 0.2756 | 0.4292 | 0.4394 |
| `044` instruct, cosine | 0.3613 ±0.0217 | 0.2606 | 0.4202 | 0.4227 |
| `probes/010` instruct, generalist harness | 0.4430 ±0.0007 | 0.3427 | 0.4988 | 0.4898 |

#### 1. The two effects do not add — they interact, and the "missing corner" is the worst cell

Cosine is worth **+0.0147** on the base backbone and **−0.0175** on the instruct one. The cell this
round was run to produce — instruct weights and the better recipe together — is the weakest of the
four. Whatever the earlier `010`-vs-`041` gap was measuring, it was not a backbone main effect that
could be stacked on top of a schedule main effect.

#### 2. The instruct deficit is a STOPPING failure, not a semantic one

The empty captions were the acute form; the chronic form survives the fix.

| arm | 4-gram repetition | predicted words | runaway (>=200 words) per seed |
|---|---|---|---|
| `041` base WSD | 0.017 | 43.2 | 12 / 21 / 4 |
| `042` base cosine | 0.018 | 43.8 | 27 / 4 / 41 |
| `043` instruct WSD | 0.035 | 48.2 | **126** / 34 / 40 |
| `044` instruct cosine | 0.040 | 46.9 | 44 / 11 / **133** |

Reference captions average 43.9 words. The instruct arms repeat at twice the rate, run long, and in
each sweep **the worst-scoring seed is the one with the most runaways** (`043` seed0 at 126 ->
0.3551; `044` seed2 at 133 -> 0.3440). That also explains the shape of the gap: METEOR falls by
0.010 where BLEU-2 falls by 0.029, because a model that says roughly the right things and then keeps
saying them loses n-gram precision long before it loses meaning. The elevated seed spread (±0.021
against the base arm's ±0.002) is the same thing — how badly a seed fails to stop is what varies.

#### 3. `010` still beats every cell here, and it is NOT budget

0.4430 against the best in-package cell's 0.4226, on the *same* instruct backbone that finishes last
in this 2x2. The obvious explanation is exposure, and it is wrong — measured rather than assumed:

| config | optimizer steps | examples/step | tokens seen |
|---|---|---|---|
| `042` base cosine | 4,884 | 64 | **120.2 M** |
| `041` / `043` / `044` | 3,663 | 64 | 90.1 M |
| `probes/010` instruct | 5,000 | `tokens_per_step` 16384 | **81.9 M** |

at a measured mean of **384.5 tokens per example** on the cap-128 build. `010` wins on the *fewest*
tokens and the fewest molecules (21,612 against 26,071). An earlier draft of this section put the
ratio the other way round, from an assumed ~200 tokens per example; the assumption was the error.

What is left is harness-level, and two of the differences are not tuning knobs but changes to the
objective and to the data:

* **Loss normalisation.** `010` sets `loss_norm: per_example`. This package has no such setting at
  all and takes HF's default per-token mean, so long captions dominate the gradient in proportion to
  their length. The stop decision is one token of ~384 either way, but the two schemes weight it
  differently across examples.
* **Truncation.** `010` sets `max_length: 512`; this package does not truncate. At a 384.5-token
  mean a real tail exceeds 512, so the two runs are not supervised on the same text.
* Batch composition (16,384 tokens/step against ~24,600) and the separate anneal fork.

**Which of these carries the result is not established.** It is the obvious next experiment and it
has not been run; nothing in the section depends on the answer.

### 6e. `loss_norm` tested and REJECTED as the explanation (2026-09-19)

§6d.3 left four harness-level candidates for `probes/010`'s 0.02 advantage, and named
`loss_norm` the leading one: it was the only candidate that changes the OBJECTIVE rather than the
trajectory, and the only one with a mechanism matching the measured failure — a per-token mean
weights an example by its length, so long captions set the gradient and the model is pushed long.

Three configs, each a ONE-VARIABLE twin of an existing three-seed cell, two seeds apiece:
`045` <-> `043`, `046` <-> `042`, `047` <-> `044`. Published protocol, benchmark denominator.

| cell | per_token | per_example | delta |
|---|---|---|---|
| instruct + WSD | 0.3788 ±0.0214 | 0.3811 ±0.0090 | **+0.0023** |
| **base + cosine** | **0.4226 ±0.0101** | 0.4039 ±0.0028 | **-0.0187** |
| instruct + cosine | 0.3613 ±0.0217 | 0.3736 ±0.0075 | **+0.0123** |

**The verdict is no.** The effect is not a main effect in either direction: it helps two cells,
hurts the best one, and nets out near zero. Nothing here closes a 0.02 gap to `010`, and the best
configuration in this package is still `042` — base weights, cosine, **per-token** loss.

#### The mechanism was real and it still did not generalise

This is the part worth keeping. The prediction — per-example removes the length bias, so generation
should land on the reference's 43.9 words — was made before the runs and came true exactly where the
pathology was worst, and *reversed* where it was mildest:

| cell | per_token (rep4 / words / runaway) | per_example |
|---|---|---|
| instruct + WSD | 0.035 / 48.2 / 66.7 | **0.031 / 43.6 / 43.5** |
| base + cosine | **0.018 / 43.8 / 24.0** | 0.033 / 45.0 / 57.5 |
| instruct + cosine | 0.040 / 46.9 / 62.7 | 0.043 / 47.3 / 76.0 |

On instruct+WSD it did what it was supposed to: 48.2 -> 43.6 words against a 43.9 reference, runaways
cut by a third. On base+cosine — the arm that was already well behaved — it made every stability
measure WORSE: repetition nearly doubled and runaways went 24 -> 58. A correction for a length bias
is only a correction where the length bias exists; applied to an arm that did not have one, it
becomes a bias of its own.

So `loss_norm` is a genuine knob for a *sick* generation arm and a pessimisation for a healthy one.
It is not the recipe change §6d.3 was hunting for.

#### One observation not to over-read

Seed spread fell in all three pairs (±0.0214 -> ±0.0090, ±0.0101 -> ±0.0028, ±0.0217 -> ±0.0075).
That is suggestive and consistent in direction, **but it is two seeds against three**: an sd from two
samples is |x1 - x2| / sqrt(2) and carries almost no confidence. It is recorded as something to check
if the knob is ever revisited, not as a finding.

#### What is left of the `010` question

Three candidates, all untested: **batch composition** (16,384 tokens/step against ~24,600), the
**separate anneal fork**, and **warmup length** (200 steps against 366/407). The field is narrowed by
one. Nothing in the paper section depends on which of the three it is.

#### 4. What the section should report

The **best defensible ChEBI-20 number this project has is `probes/010` at 0.4430 ±0.0007**, and the
best in-package number is `042` at 0.4226 ±0.0101. Both sit **below InstructMol-G's 0.466** (§5),
which is the direct architectural comparator at 7x the parameters. The section says that plainly
rather than choosing a friendlier opponent afterwards.

The 2x2 is reported as methodology, and it retires two claims this file used to make:

* "The instruct base helps a lot" — **false as stated.** It was an artifact of comparing a
  never-annealed control (§6c) against a run that differed in four ways at once, and of a generation
  defect that cost the instruct arm 6-24 % of its captions.
* "Cosine beats WSD" — **true only on the base backbone**, and even there the +0.0147 includes about
  +0.004 of dev-selection the WSD arm does not get.

## 7. The section

Every number below traces to a file in this repository: the property rows to `results/*/runs.jsonl`
(§6, S11), the ChEBI-20 row to `src/generalist/results/chebi/lit_metrics.json` (§6, S4/S6), the
anchors to §5. Three seeds per cell throughout.

```latex
\subsection{Molecules}
\label{sec:molecules}

We evaluate GTLM as a per-task specialist on the two molecule benchmark families
that admit a published comparison: scaffold-split property prediction on
MoleculeNet (BACE, BBBP, HIV) and molecule captioning on ChEBI-20. Every cell is
three seeds on a 1B Llama backbone with LoRA; we report mean $\pm$ standard
deviation over seeds. The graph arm reads the molecular graph as prefix nodes
under a structural attention bias and never sees a SMILES string, so every row
below is a graph-only result.

\begin{table}[t]
\centering\small
\caption{MoleculeNet property prediction, scaffold split, ROC-AUC ($\uparrow$).
Baselines as compiled in InstructMol's Table~2; $\dagger$ marks rows run by those
authors.}
\label{tab:molnet}
\begin{tabular}{lcccr}
\toprule
 & BACE & BBBP & HIV & params \\
\midrule
Uni-Mol (3D conformers)          & 85.7 & 72.9 & 80.8 & --- \\
MolCA (1D+2D)                    & 79.8 & 70.0 & ---  & 1.3B \\
GraphCL (structure only)         & 75.3 & 69.7 & 78.5 & --- \\
ChemBERTa-2 (77M SMILES)         & 73.5 & 69.8 & 79.3 & --- \\
\midrule
InstructMol-G$^\dagger$          & 84.3 & 68.6 & 74.0 & 7B \\
Llama-2-7B-chat + LoRA$^\dagger$ & 74.8 & 65.6 & 62.3 & 7B \\
Vicuna-v1.3-7B + LoRA$^\dagger$  & 68.3 & 60.1 & 58.1 & 7B \\
\midrule
\textbf{GTLM (graph)}       & $82.0\pm1.2$ & $70.6\pm2.3$ & $76.9\pm1.7$ & 1B \\
\textbf{GTLM (SMILES twin)} & $82.2\pm2.5$ & $71.6\pm2.2$ & $76.2\pm1.3$ & 1B \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering\small
\caption{ChEBI-20 molecule captioning, scored on the full 3{,}300-molecule test
split under the reference metric protocol. ``graph in'' marks systems whose input
is a molecular graph rather than a SMILES string.}
\label{tab:chebi}
\begin{tabular}{lccccr}
\toprule
 & BLEU-2 & BLEU-4 & ROUGE-L & METEOR & params \\
\midrule
MolT5-Base            & 0.540 & 0.457 & 0.568 & 0.569 & 250M \\
MolT5-Large           & 0.594 & 0.508 & 0.594 & 0.614 & 780M \\
Text+Chem T5-augm     & 0.625 & 0.542 & 0.622 & 0.648 & 250M \\
MolCA (Galac-1.3B)    & 0.620 & 0.531 & 0.618 & 0.651 & 1.3B \\
GPT-4-0314 (10-shot)  & 0.607 & 0.525 & 0.562 & 0.610 & --- \\
\midrule
Mol-Instruction          & 0.249 & 0.171 & 0.289 & 0.271 & 7B \\
BioMedGPT-10B            & 0.234 & 0.141 & 0.332 & 0.308 & 10B \\
InstructMol-G (graph in) & 0.466 & 0.365 & 0.479 & 0.491 & 7B \\
\midrule
\textbf{GTLM (graph in)} & $0.443$ & $0.343$ & $\mathbf{0.499}$ & $0.490$ & 1B \\
       & \tiny$\pm0.001$ & \tiny$\pm0.001$ & \tiny$\pm0.001$ & \tiny$\pm0.002$ & \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Property prediction.}
Both arms place sixth to seventh of seventeen on all three sets: below the 3D and
multi-view pretrained specialists, level with 2D-pretrained GNNs, and above every
general-purpose-LLM row at one seventh of those rows' parameters. We report the
SMILES twin beside the graph arm because the two are statistically
indistinguishable---pooled over nine paired seeds the difference is $-0.0016$,
95\% CI $[-0.023,+0.020]$---and quoting the graph arm alone would invite the
reader to credit the structural channel with a result the matched control does
not support. Per dataset, three seeds resolve only differences above $0.077$, so
the per-set signs should not be read.

\paragraph{Captioning.}
Against InstructMol-G---the directly comparable system, being graph-only input
with LoRA adaptation---GTLM is ahead on ROUGE-L ($+0.020$), level on METEOR
($-0.001$) and modestly behind on BLEU-2/4 ($-0.023$, $-0.022$), at one seventh
of the parameters; it leads the remaining LLM rows by $0.19$--$0.21$ BLEU-2. The
chemistry-pretrained sequence models remain well ahead and we do not claim
against them: they read SMILES and are pretrained on chemical corpora, while this
model reads a graph and starts from a general-purpose 1B backbone.

\paragraph{Protocol.}
ChEBI-20 is scored with the reference implementation's metrics: NLTK corpus BLEU
and \textsc{Meteor} over SciBERT word pieces and \texttt{rouge\_score} over raw
strings. We score the \emph{whole} 3{,}300-molecule test split; the 39 molecules
above our 128-heavy-atom encoding limit are charged as empty predictions rather
than dropped, so the denominator is the benchmark's. Restricting to the admitted
3{,}261 would raise BLEU-2 by $0.011$. We state this because the practice is not
universal and the effect need not be small: on an earlier build admitting 2{,}889
of 3{,}300, the same charge was worth $0.068$ BLEU-2.

\paragraph{Recipe and backbone.}
We ran the captioning task as a $2\times2$ over backbone (base vs.\ instruction-tuned
weights, each with its matching prompt format) and schedule (warmup-stable-decay
vs.\ a single-run cosine at a $2\times$ learning rate and a $1.3\times$ budget),
three seeds per cell, on ChEBI-20's own split. The two factors do not compose. On
the base backbone the cosine recipe is worth $+0.015$ BLEU-2 ($0.423 \pm 0.010$
against $0.408 \pm 0.002$); on instruction-tuned weights it is worth $-0.018$
($0.361 \pm 0.022$ against $0.379 \pm 0.021$), so the cell combining both is the
weakest of the four. \textbf{The deficit is a stopping failure rather than a
semantic one}: the instruction-tuned arms repeat $4$-grams at twice the base rate,
exceed $200$ words on $3$--$10\times$ as many molecules, and in each sweep the
worst seed is the one that runs away most often. METEOR falls by $0.010$ where
BLEU-2 falls by $0.029$, which is the signature of a model that says roughly the
right thing and then continues past the end of it.

\paragraph{A note on generation.}
Captions are decoded greedily with the stop token suppressed for the first five
steps, uniformly across arms. Without it, an immediately emitted end-of-turn token
yields an empty string that is scored as a total miss while perturbing validation
loss by under $0.005$; on instruction-tuned weights, whose chat template makes an
empty assistant turn a natural continuation, this silently discarded $6$--$24\%$ of
captions. The constraint is inert on the base backbone, which emits no empty
captions either way.

\paragraph{Scope.}
We report classification and captioning. MoleculeNet's regression sets (ESOL,
FreeSolv, Lipophilicity) fall outside a readout that scores a yes/no logit margin
and cannot emit a number. Tox21 and SIDER are trained on but not reported: their
per-endpoint AUROC has no comparable ladder against LLM baselines. ClinTox is
held out of every training mixture throughout this work.
```

### Notes for the author, not for the section

* **The captioning row is a specialist and the property rows are specialists, but not the same
  recipe.** Property rows: base `Llama-3.2-1B`, `Q:/A:` formatting, single-run cosine through
  `molecules/train.py`. Caption row: `Llama-3.2-1B-Instruct`, chat formatting, WSD plus a 500-step
  cosine anneal through the generalist harness, because ChEBI-20 exists only there (§4). Both seams
  are disclosed in §4; neither is hidden by the section, which says "a 1B Llama backbone with LoRA"
  and nothing stronger. If a reviewer asks, the answer is that the tasks are scored on different
  metrics and were never one protocol.
* **`\pm` on the caption row is 0.001, which will look implausible.** It is real: three seeds on a
  3,300-molecule generative split, per-seed BLEU-2 of 0.4422 / 0.4436 / 0.4431. Consider giving the
  per-seed values in an appendix rather than inviting the suspicion.
* **Two numbers in Table~\ref{tab:molnet} are the campaign's own closing values and carry §8.4.8's
  caveat**: the graph/flat difference is not resolvable at three seeds per dataset. The text says so.
* **The generalist comparison is deliberately absent.** +0.114 BLEU-2 for specialising over the
  20 %-share generalist (§6) is a good result and belongs in the architecture section, not in a
  table whose premise is comparability with published rows.
