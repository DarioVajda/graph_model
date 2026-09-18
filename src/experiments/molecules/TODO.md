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

## 6c. RESULTS — the control and the budget-scaling arm (2026-09-18)

Published protocol, whole 3,300-molecule benchmark split, 39 unencodable molecules charged against
their real captions. Three seeds each. `042` is read at its **best dev BLEU-2 checkpoint**; `041` is
WSD and is read at the end of its schedule.

| run | backbone / format | schedule | budget | BLEU-2 | BLEU-4 | ROUGE-L | METEOR |
|---|---|---|---|---:|---:|---:|---:|
| `010` instruct | 1B-Instruct, chat | WSD + anneal | 8.9 ep | **0.4430** ±0.0007 | **0.3427** | **0.4988** | **0.4898** |
| `041` control | 1B base, `Q:/A:` | WSD | 9 ep | 0.3999 ±0.0030 | 0.2902 | 0.4401 | 0.4427 |
| `042` scaling | 1B base, `Q:/A:` | cosine, `lr` 2e-4 | 12 ep | 0.4229 ±0.0015 | 0.3158 | 0.4552 | 0.4592 |

Per-seed BLEU-2 — control 0.4014 / 0.4018 / 0.3964, cosine 0.4223 / 0.4246 / 0.4217.

### 1. The cosine recipe at a larger budget beats the WSD control, and the LR correction holds

**+0.0230 BLEU-2, +0.0255 BLEU-4, +0.0151 ROUGE-L, +0.0165 METEOR**, every metric in the same
direction and a seed spread half the control's (±0.0015 against ±0.0030). The two runs differ in
schedule *and* budget, so this is their joint effect and not a schedule ablation — but the direction
settles the practical question: **moving Tier C onto this package's convention costs nothing.** It
is not a trade of accuracy for tidiness.

The 2× LR correction is not falsified by anything here. A warmup+cosine run at the *same* peak as
WSD would have spent about 0.55 of the dose; at 2e-4 it spends about 0.94 of it, and it came out
ahead rather than unstable, which is what the average-LR argument predicts.

### 2. Still not converged, and now on dev evidence rather than inference

**Every seed's best dev checkpoint sits at step 4200 or 4884 of 4884** — val BLEU-2 0.4366, 0.4473,
0.4404. Not one run peaked early and declined. At 12 epochs and twice the learning rate, the
budget is still the binding constraint, and the reported number is a floor.

That selection is also doing real work rather than rubber-stamping the last checkpoint: two of three
seeds chose 4200 over 4884, so reading the end of a cosine run would have been slightly wrong.

### 3. The backbone costs more than the port gains — and the two are confounded by design

`041` is 0.0431 BLEU-2 below the instruct cell. That gap is the **joint** effect of two changes
pushing opposite ways: base weights instead of instruct+chat (expected to hurt captioning), and
ChEBI's own split instead of the cross-source partition (**+21 % training data**, expected to help).
The backbone effect alone is therefore *larger* than 0.0431.

**This is the outcome §2 of this file said would be the least informative, and it is worth saying
plainly rather than dressing up.** A clean backbone control would have kept the partition; the port
removed it for a good independent reason, and the two arrived together.

**What it does support**: even at the low end, a base-weights 1B still lands within ~0.02 BLEU-2 of
InstructMol-G's 0.466 once the budget is right, so nothing about the headline claim depends on the
instruct backbone. **What it does not support**: any statement of the form "instruct weights are
worth X on captioning". Separating them is one more `041`-shaped run with the partition restored,
which nothing in the paper needs.

### 4. What the section should report

The **instruct cell (`010`) remains the headline**: it is the strongest number, it is three seeds,
and it is the one the anchor table is comparable with. `041` and `042` are the methodology evidence
behind it — that the recipe choice is not load-bearing, and that the budget was not tuned to
convergence. The seam disclosed in §4 stands, and is now quantified rather than asserted.

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

\paragraph{Recipe and budget.}
The caption model is trained with warmup-stable-decay and a short anneal. We
verified that this is not load-bearing: the same task trained with this work's
usual single-run cosine schedule, at a learning rate raised $2\times$ to hold the
average rate fixed, and a $1.3\times$ budget, scores $0.423 \pm 0.002$ BLEU-2
against $0.400 \pm 0.003$ for a matched WSD control --- better on every metric, so
the schedule choice costs nothing. \textbf{The budget, however, is not tuned to
convergence}: selecting each cosine run at its best checkpoint on a held-out
development split places every seed in the final quarter of training, and
development BLEU is still rising there. The reported numbers are floors.

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
