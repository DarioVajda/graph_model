# The molecule generalist campaign — what the twelve cells found

`molecule_generalist.jsonc` beside this file is the campaign: one 1B model
(Llama-3.2-1B + LoRA) over every molecule task, trained four ways at one recipe,
three seeds each. This is what came out. The full argument, the mixture design
and the history live in `../../MOLECULE_GENERALIST.md`; this file is the results
table, so that a number can be looked up without reading the case for it.

**Two scopes to know before quoting anything here.** This campaign is the
*notation ladder*, on base weights at the 1× horizon; the reportable molecule
generalist is now the instruct campaign in `../probes/008_*.jsonc`, written up at
`../../MOLECULE_GENERALIST.md` §9 Tier 2c. And the **Generation** table below was
measured through the stop-token defect found on 2026-09-10 (§8 there): those rows
score whether the model stopped, not whether it was right, and the conclusion
drawn from them at the time has been withdrawn in place.

| arm | what the model sees |
|---|---|
| `graph` | the Levi graph — atoms and bonds as nodes, topology as SPD + magnetic attention biases |
| `flat` | the same molecule as a **SMILES** string, one node |
| `flat_selfies` | the same molecule as **SELFIES** |
| `flat_inchi` | the same molecule as **InChI** |

All four determine the same molecule, so a difference between the three flat arms
cannot be an information difference. What separates them is how much of each
notation the backbone read during pretraining — which is the whole reason the
last two exist.

Every number below is the **annealed** checkpoint at step 6160, averaged over
seeds 0/1/2 and quoted as mean ± sd. A trunk stops mid-WSD-stable-phase by
construction, so its last checkpoint is not the model any comparison should use;
the reportable checkpoint is the anneal fork's
(`../forks/anneal_molecule_generalist.jsonc`), and it is a separate submission per
cell.

---

## The three findings

1. **The flat arm's property-prediction lead is SMILES-specific, and it
   reverses.** Against the graph arm: SMILES **+0.0135** ROC-AUC, InChI
   **−0.0138**, SELFIES **−0.0167**. The sign holds in **3/3 seeds** for all
   three comparisons.
2. **The graph arm wins structure, and wins it hardest where nothing was
   trained.** It takes 8 of 11 exact-match probes, and its two largest margins
   are the two *held-out* topology tasks — 2.2× and 1.6× the best flat arm.
3. **None of it is a pretrained prior.** Untrained, no arm on any set is usefully
   above chance; the base model's zero-shot ordering is not the ordering the
   trained models produce, and would have inverted the selection rule that
   motivated the ladder.

---

## Tier 0.1 — the untrained base model

Before training anything, the same five property sets scored on
**Llama-3.2-1B with no adapters at all** — a yes/no logit margin read off the
frozen backbone (`tools/notation_probe.py`, no `--checkpoint`). The graph arm here
is the architecture with untrained biases and untrained LoRA, which is the honest
floor for it: whatever the graph arm knows, it did not bring.

ROC-AUC, 500 rows a set, truncated rows excluded from every arm:

| set | graph | SMILES | SELFIES | InChI |
|---|---|---|---|---|
| BACE | 0.5587 | 0.2649 | 0.5410 | 0.4420 |
| BBBP | 0.4577 | 0.5456 | 0.4769 | 0.4271 |
| HIV | 0.6343 | 0.3294 | 0.6624 | 0.5323 |
| SIDER | 0.5325 | 0.4752 | 0.5219 | 0.4732 |
| Tox21 | 0.4352 | 0.4389 | 0.3869 | 0.4818 |
| **mean** | **0.5237** | **0.4108** | **0.5178** | **0.4713** |

**No arm is usefully above chance on any set.** BACE/SMILES at 0.2649 and
HIV/SMILES at 0.3294 are not weak signal — they are *anti*-signal, a model
answering the wrong way round with confidence. Nothing here predicts what the
trained ordering will be.

**This is why no notation was selected.** The ladder's stated design was to
identify the notation carrying least pretraining bias and compare against that.
On this data the rule is incoherent: **SMILES scores the lowest mean AUROC
(0.4108) while carrying the most prior signal of the four arms.** Measuring prior
as distance from chance rather than as score,

| | graph | SMILES | SELFIES | InChI |
|---|---|---|---|---|
| mean \|AUROC − 0.5\| | 0.0665 | **0.1074** | 0.0723 | **0.0416** |

lowest score and least prior are opposite orderings, because SMILES's errors are
systematically anti-predictive rather than random. So all three notations are
reported and none is a baseline — picking one would have been baseline-shopping
dressed as a control.

Two structural readings worth keeping. The flat arms answer from a handful of
distinct margins (5–15 across a whole set) against the graph arm's 39–51, and
their tied-pair fractions run 0.13–0.35 against the graph arm's 0.03–0.04: an
untrained flat arm is close to emitting a constant. And the graph arm's mean of
0.5237 is the number that matters for the campaign's claim — it is chance, so the
graph arm's trained performance is *acquired*, not inherited.

---

## Property classification — the trained models

Scored on one clean instrument (`tools/notation_probe.py --checkpoint`), with the
**union** of truncated rows excluded from all four arms so that every arm scores
the identical row set. ROC-AUC, mean ± sd over three seeds:

| set | graph | SMILES | SELFIES | InChI |
|---|---|---|---|---|
| BACE | 0.8185 ±0.0193 | **0.8667** ±0.0101 | 0.8229 ±0.0143 | 0.7888 ±0.0211 |
| BBBP | 0.7072 ±0.0123 | **0.7086** ±0.0138 | 0.6940 ±0.0114 | 0.6909 ±0.0079 |
| HIV | **0.7374** ±0.0064 | 0.7291 ±0.0095 | 0.6896 ±0.0192 | 0.7372 ±0.0391 |
| SIDER | 0.8332 ±0.0078 | **0.8468** ±0.0112 | 0.8358 ±0.0076 | 0.8351 ±0.0005 |
| Tox21 | 0.7971 ±0.0070 | **0.8100** ±0.0156 | 0.7677 ±0.0066 | 0.7724 ±0.0185 |
| **five-set mean** | **0.7787** ±0.0054 | **0.7922** ±0.0085 | **0.7620** ±0.0056 | **0.7649** ±0.0066 |
| **vs graph** | — | **+0.0135** | **−0.0167** | **−0.0138** |

The three gaps are each about 1.6–3× the per-seed spread, which on its own would
be suggestive rather than settled. What makes it more than suggestive is that the
comparison is **paired by seed and the sign never flips**: seed by seed the
five-set means are

| seed | graph | SMILES | SELFIES | InChI |
|---|---|---|---|---|
| 0 | 0.7828 | 0.8019 | 0.7600 | 0.7649 |
| 1 | 0.7806 | 0.7861 | 0.7684 | 0.7583 |
| 2 | 0.7726 | 0.7886 | 0.7577 | 0.7715 |

SMILES beats the graph arm in 3/3, and each of SELFIES and InChI loses to it in
3/3. **The advantage is a property of the notation, not of flatness.**

### The same sets on the training harness's own instrument

The harness scored these during the run without excluding truncated rows, giving
graph 0.7796 / SMILES 0.7969 / SELFIES 0.7634 / InChI 0.7697. The two instruments
agree to within 0.003 everywhere **except SIDER**, where the harness reads
−0.011 (SMILES), −0.013 (InChI) and **+0.006** (graph). That split is the
truncation defect's fingerprint and is why the clean instrument is the one quoted
above: `max_length` 512 cuts a flat prompt from the right, taking the trailing
answer with it, and the graph arm is immune because `max_length` is per node.
Truncated SIDER rows: SMILES 162, SELFIES 297, InChI 270, **graph 0** of 3,861.

### ClinTox, the held-out property set

| | graph | SMILES | SELFIES | InChI |
|---|---|---|---|---|
| ROC-AUC | 0.1942 ±0.0505 | 0.1831 ±0.0459 | 0.1737 ±0.0508 | 0.1949 ±0.0427 |

Every arm is far *below* chance, and this is not a ranking — it is a
transfer failure that all four share. ClinTox's two endpoints disagree about what
"yes" means (FDA-approved is 95 % positive; CT_TOX is 6 %), and a model that
learned the mixture's answer convention answers the wrong one consistently. It is
reported because omitting a set that embarrasses every arm equally is how a
results table stops being one.

---

## Structural probes — exact-match accuracy

| task | graph | SMILES | SELFIES | InChI |
|---|---|---|---|---|
| aromatic_ring | **1.0000** ±0.0000 | **1.0000** ±0.0000 | **1.0000** ±0.0000 | **1.0000** ±0.0000 |
| ring_membership | **0.9993** ±0.0012 | 0.9933 ±0.0042 | 0.9840 ±0.0020 | 0.9907 ±0.0031 |
| fg_atom_membership | **0.9913** ±0.0031 | 0.9467 ±0.0110 | 0.9493 ±0.0076 | 0.9373 ±0.0081 |
| fg_presence | **0.9813** ±0.0133 | 0.9673 ±0.0081 | 0.9480 ±0.0020 | 0.9007 ±0.0070 |
| stereo_assigned | 0.9733 ±0.0012 | 0.9880 ±0.0040 | **0.9900** ±0.0040 | 0.9827 ±0.0042 |
| ring_size | **0.9293** ±0.0090 | 0.8647 ±0.0244 | 0.8480 ±0.0262 | 0.8547 ±0.0061 |
| fg_count | 0.9260 ±0.0080 | **0.9647** ±0.0050 | 0.9340 ±0.0035 | 0.8840 ±0.0035 |
| ring_count | 0.8767 ±0.0103 | 0.9100 ±0.0060 | **0.9313** ±0.0099 | 0.8753 ±0.0114 |
| stereo_potential | **0.7900** ±0.0151 | 0.7887 ±0.0358 | 0.6707 ±0.0122 | 0.7827 ±0.0114 |
| longest_chain *(held out)* | **0.1013** ±0.0323 | 0.0460 ±0.0106 | 0.0340 ±0.0156 | 0.0560 ±0.0300 |
| bond_path *(held out)* | **0.0667** ±0.0061 | 0.0420 ±0.0053 | 0.0320 ±0.0171 | 0.0433 ±0.0070 |

The graph arm takes **8 of 11**. Read the bottom two rows first: `longest_chain`
and `bond_path` are the only tasks in the suite that were never trained on, they
are the two that ask for a path through the molecule, and the graph arm is 2.2×
and 1.6× the best flat arm on them. Everything above them is above 87 % for
every arm and is mostly saturated.

The three the graph arm loses — `fg_count`, `ring_count`, `stereo_assigned` — are
counting and flag tasks, where a linear string is a perfectly adequate substrate
and the topology buys nothing. **Four of these families (`ring_membership`,
`aromatic_ring`, `ring_size`, `fg_atom_membership`) are in SMILES on every arm**,
because they ask about a *named* atom and only SMILES can mark one; the notation
columns there are not measuring notation.

---

## Generation

| task | metric | graph | SMILES | SELFIES | InChI |
|---|---|---|---|---|---|
| ChEBI-20 | BLEU-2 | 0.1937 ±0.0031 | **0.2004** ±0.0045 | 0.1965 ±0.0065 | 0.1930 ±0.0024 |
| ChEBI-20 | BLEU-4 | 0.1324 ±0.0028 | **0.1391** ±0.0042 | 0.1359 ±0.0044 | 0.1321 ±0.0026 |
| ChEBI-20 | ROUGE-L | 0.3059 ±0.0034 | **0.3146** ±0.0048 | 0.3101 ±0.0071 | 0.3047 ±0.0030 |
| ChEBI-20 | METEOR | 0.4573 ±0.0044 | **0.4711** ±0.0057 | 0.4673 ±0.0089 | 0.4566 ±0.0030 |
| g2s | validity | 0.0560 ±0.0069 | 0.1527 ±0.0540 | **0.2093** ±0.0397 | 0.1347 ±0.0463 |
| g2s | exact_match | 0.0000 ±0.0000 | 0.0193 ±0.0145 | **0.0700** ±0.0300 | 0.0007 ±0.0012 |
| g2s | roundtrip_match | 0.0000 ±0.0000 | 0.0300 ±0.0173 | **0.1180** ±0.0262 | 0.0007 ±0.0012 |

**ChEBI-20 is a four-way tie.** The spread across arms is 0.007–0.015 against a
seed spread of 0.003–0.009; captioning is 20 % of the mixture and no arm gets
anything out of the others' representation.

~~**g2s is not a tie, and it is the graph arm's clear failure.**~~ **Withdrawn
2026-09-10 — it was an instrument defect, and the sign was backwards.** The
reading was that the graph arm produces a valid string 5.6 % of the time and a
*correct* one zero times in 1,500 attempts, so reading a graph and writing a
string are different skills and this campaign only trained one of them. No
generative answer in this build carried a stop token, so what the table above
scores is whether a model happened to stop: the graph arm emits the exactly
correct canonical SMILES as a **prefix** of its output 46.5 % of the time and then
runs on to the generation cap. Retrained with the stop token on, it is exactly
right **41.9 %** of the time against the SMILES arm's 21.1 %
(`../../MOLECULE_GENERALIST.md` §9 Tier 2c). The rest of the paragraph still
holds: SELFIES leads on `validity` by construction rather than by merit — nearly
every SELFIES string decodes, so that column measures the grammar — and InChI is
at floor because that arm must produce a SMILES answer from an InChI-conditioned
prompt.

---

## What the whole suite says

The three families disagree, consistently, and the disagreement is the result:

* **Property prediction** — SMILES leads, and only SMILES. Change the notation
  and the flat arm drops below the graph arm by about the margin it led by. The
  lead is a pretraining prior, not an advantage of linear representation.
* **Structure** — the graph arm leads, and leads by most on the two tasks it was
  never trained on. That is the direction a representation claim wants to point.
* **Generation** — SMILES leads on captioning by a margin inside the noise. The
  second half of this line used to read "and the graph arm cannot write a
  molecule at all"; it was the instrument, and retrained with a stop token the
  graph arm writes molecules — and writes them better than its SMILES twin at the
  same recipe.

None of which is a scale claim. Every cell is 1B parameters at 5,599 steps, and
nothing here says what happens at 3B or at 20,000.

---

## Caveats that travel with every number

1. **Truncation is arm-asymmetric, was not fixed, and is small.** `max_length` 512
   truncates a flat prompt from the right, taking the trailing answer with it;
   `render` then supervises the prompt node's last token, which on such a row is a
   mid-molecule token (`sider/flat` row 324 is supervised on `@@` against a stored
   answer of ` Yes`). The graph arm is immune, because the cap is per node and an
   atom text is a handful of tokens. Evaluation excludes truncated rows from every
   arm; **training cannot**, so the flat arms trained on some corrupted supervision
   and the graph arm did not — and the longer notations got more of it, which is
   the same direction as the reversal reported above.

   `tools/truncation_census.py` counts every training row against the cap,
   weighted by mixture share:

   | | graph | SMILES | SELFIES | InChI |
   |---|--:|--:|--:|--:|
   | share of the training draw truncated | **0.000 %** | 0.165 % | 0.521 % | 0.332 % |

   **SIDER is essentially the whole of it** — 189 / 15,390 rows on SMILES against
   572 on SELFIES and 405 on InChI — and it is one of five sets, so its effect on
   the five-set mean is diluted fivefold. ChEBI-20 and g2s were the two sources
   worth worrying about, since their answers are a full caption and a full SMILES
   string competing with the molecule for the same 512 tokens; measured, they are
   clean (26 / 20,477 and 10 / 4,000, SELFIES only). The four Tier-A families held
   at SMILES on every arm truncate *identically* across the flat arms — 15, 21, 9
   and 7 rows — so they cannot differentiate them, which is the disclosure below
   working as intended.

   A ~0.5 % label-noise differential concentrated in one of five sets cannot
   plausibly account for a 0.0167 gap, so the reversal survives it. Raising
   `max_length` and re-running is still what would settle it outright, and it is
   not worth the compute at this size.
2. **~11 % of the mixture is identical on all three flat arms.** The four
   atom-naming Tier-A families stay in SMILES everywhere, because only SMILES can
   mark an atom. This keeps the mixture identical across arms — same families,
   same shares, same rows — at the cost of diluting the notation contrast by that
   much.
3. **InChI merges tautomers**, so it is not strictly bijective with the molecule
   the way SELFIES is, and **its formula layer hands over the composition**
   (`InChI=1S/C12H15NO4/…`) where SMILES makes the model derive it. Both cut in
   InChI's favour on any task a formula helps with; InChI still lost.
4. **Tox21 and SIDER are not anchor-comparable** to the published specialist
   numbers, and ClinTox transfers to nothing. The five-set mean is a within-
   campaign quantity.
5. **SELFIES needed a widened constraint set** (`hypervalent` with the catch-all
   raised to 12) to encode 22 organometallics of 53,921. InChI needed none.

---

## The recipe, and why

`molecule_generalist.jsonc` states what was run; this section is the argument
behind each value in it. Twelve cells in one file rather than twelve files:
twelve copies that must stay identical everywhere except a handful of fields is
twelve chances to edit one and not the others, and a recipe that drifts in one
cell of a twelve-cell comparison reads as a seed effect rather than as the
mistake it is. Here the invariant is structural — the shared recipe exists once —
instead of being a test that checks twelve files against each other.

### What varies, and why each one has to

**`arm` and `bias`.** `RunConfig.validate` refuses a flat arm carrying a bias: on
a single-node graph every structural bias is identically zero (Property 2), so a
bias string in the record would advertise a comparison that is not happening.
`bias_lr` stays at the graph arm's 1e-2 on every arm — it reaches no tensor
without a bias channel, so holding it equal keeps the cells differing in `arm`
alone, which is what the record should say.

**`tokens_per_step`.** D4.4 sets the batch in tokens so that a mixture of very
differently-sized tasks costs a roughly constant amount per step — the right
default, and exactly wrong for an arm comparison. **The arms are matched in
examples, not tokens**, because a gradient is made of examples. The built mixture
measures 288.28 tokens an example on the graph arm against the SMILES arm's
82.51, with SELFIES and InChI between them; sharing one token budget would hand
the graph arm 3.5× the batch and the comparison would stop being a comparison.
Each value is `tools/tokens_per_step.py` run against that arm's own measured
`mean_tokens` at the same 56.83 examples/step.

The SMILES arm's 4689 rather than 4690 is worth stating: at `CORPUS_PASSES` 6 no
integer lands it on the graph arm's 5,599 steps (4689 gives 5600, 4690 gives
5598), so `max_steps` pins the horizon for every arm and the budget is chosen on
the short side of it — 5,599 steps draws 318,179 examples against a budget of
318,217, where 4690 would ask for more than the budget holds. The residual is
0.012 % on examples per step, against a step count that now matches exactly.

**`accumulation_steps`, with `gpus_per_config`.** The memory knob and nothing
else (D4.4): `micro_batch_tokens` is `tokens_per_step / (accumulation_steps ×
world_size)`, so cells take the same examples in the same order and produce the
same optimizer step whatever it is set to. It is per arm because the arms have
genuinely different memory profiles. The graph arm needs 16: at 8 it is a
2048-token micro-batch, and the first launch died at step 20 with `Tried to
allocate 16.78 GiB … 151.22 GiB is allocated by PyTorch` on a 178 GB B200. 16
puts it back to the 1024 the shakedown ran at and measures 100.6 GB at step 200,
which is also why the GPU list cannot include an 80 GB H100 at any
`accumulation_steps`.

That failure was **stochastic** — seed 1 was still healthy at step 93 when seed 0
had already died — because the peak is set by the longest molecule that happens
to land in a micro-batch. A cell surviving its first hundred steps proves
nothing; the headroom has to be read off the peak, not off whether a run is still
alive.

The notation arms carry `gpus_per_config: 2` with `accumulation_steps: 4`, which
is the same micro-batch and therefore the same optimizer step, bought for half
the wall clock. They can afford DDP width and the graph arm cannot: `DESIGN.md`
§D9 records that it cost the *graph* anneal 96 % of its wall clock in Triton
autotuning, and the flat arms recompile just as often but tune far more cheaply.
Both fields are spelled per cell rather than in `execution`, because that block
belongs to the sweep runner and cannot vary per cell; `chain.sh` takes them from
the resolved config, so the file submits the form that actually ran.

**`validators`.** The notation arms drop `perm_spread`, which rewrites a
molecule's atom order in place and so needs a notation that can express a
non-canonical ordering. SELFIES and InChI are canonical-only, and
`flat_serialize` refuses rather than handing back a silently canonical string
that would report a floor of zero as if it had been measured.

**`seed`.** Initialisation and example order only. `data_seed` stays 0 on every
cell, so all twelve read one build and a difference between two seeds is the
run's own spread and nothing else. That is the whole point of three seeds —
`../../MOLECULE_GENERALIST.md` §8.4 has two cases of a single seed telling this
campaign something that did not replicate.

Nothing else may differ, and
`tests/generalist/test_cli.py::test_the_campaign_cells_differ_only_where_they_are_meant_to`
asserts it field by field.

### The numbers held fixed

**`lr` 1e-4, matched across arms** (settled 2026-09-04). The specialists settled
the rate per (task, arm) — 3e-4 on BACE and BBBP, 1e-4 on HIV — and one mixture
cannot hold both, so the choice is which a single rate inherits. The schedule
argues for the lower one: the specialists ran warmup + cosine, which touches its
peak for a moment and spends most of the run below it, while WSD holds the stable
phase at `lr` for essentially the whole run, so the same number is a materially
larger dose here. So does the asymmetry of the risk: 3e-4 was measured to cost
the graph arm 0.109 ROC-AUC on HIV, where it also produced the screen's highest
validation and lowest test score, while 1e-4 on BACE and BBBP was screened and
came out worse rather than broken. HIV is 10.6 % of this mixture against BACE's
2.0 % and BBBP's 2.4 %.

**`max_steps` 5599, pinned** rather than left to the §2 budget rule, and pinned to
the number that rule gives the graph arm at `CORPUS_PASSES` 6. The horizon has to
be identical across arms and the budget rule cannot deliver that on its own,
because the arms match on examples per step rather than tokens per step.

**`max_spd` 32.** The clamp ablation is null — the governing quantity is the
fraction of node *pairs* past the ceiling, 2.56 % on BACE and 1.20 % on BBBP, not
the 53 % of molecules an earlier reading quoted — and 32 is what every molecules
sweep from `001` to `028` used, so it keeps this campaign comparable with the
specialists.

**`mem` 128G, measured rather than inherited.** `molecules/PLAN.md` §8.4.9's HIV
OOM was not a dataset that did not fit: at `num_workers > 0` HF's evaluation
loader is rebuilt by accelerate on every `evaluate` call, and each rebuild forks a
fresh set of persistent workers without shutting the previous set down, so a graph
cell leaked ~1 GB of unreclaimable anon per evaluation. Neither half of that can
happen here — this trainer forces `dataloader_num_workers=0` and runs no HF
evaluation loop at all (the D7 validators are the evaluation). Measured on one HIV
graph cell: 50.7 GB cgroup peak, of which 6.8 GB is anon and the rest reclaimable
page cache, dead flat across the run. 192G would be carrying a workaround for a
leak this run cannot have.

**`time` 12:00:00 and `chunks` 3.** `--time` is the window, not the workload: a
chunk is however much of the run fits before the wall clock runs out, never a
guess at how long training *should* take. Chunk 1 should finish a cell outright
and the other two are the safety net for a chunk Slurm kills; `save_steps` 500 is
what a killed chunk costs, since the chain resumes from the last complete
checkpoint.

## Reproducing it

```bash
# the twelve cell names
python3 -m src.generalist validate --config src/generalist/configs/runs/molecule_generalist.jsonc --cells

# one cell, end to end
CFG=src/generalist/configs/runs/molecule_generalist.jsonc
CELL=molecule_generalist_graph_s0
python3 -m src.generalist data_prep --config $CFG --cell $CELL
src/generalist/tools/chain.sh $CFG $CELL
GPU=1 INDUCTOR_CACHE=.inductor_cache/generalist src/generalist/tools/run_cli.sh fork \
  --from src/generalist/results/runs/$CELL/checkpoint-5599 --mode anneal \
  --fork-config src/generalist/configs/forks/anneal_molecule_generalist.jsonc \
  --config $CFG --cell $CELL

# the property tables, on one instrument across all four arms
python3 -m src.generalist.tools.notation_probe --checkpoint <anneal checkpoint>
```

All twelve cells read one build, `42f7a14bed21f876`, so a difference between two
cells is the run's own spread and nothing else. Each also resolves to the
`config_hash` its run record carries — `test_the_campaign_still_resolves_to_the_runs_that_were_measured`
pins all twelve, because an edit that changes what a cell resolves to would
silently detach this write-up from the runs it describes.

**Cost, measured after the `DESIGN.md` §D9 evaluation fixes:** a notation cell is
~2.8 GPU-h — 5,599 steps in about 62 minutes at two ranks including both
milestones, plus ~20 minutes annealing — and six cells ran concurrently on twelve
B200s in 1h27 wall, ~18 GPU-h. Scoring a trained checkpoint is another ~0.2
GPU-h. Arm 2's older figures (~20 GPU-h a graph cell, ~7 a flat one) predate
those fixes and overstate a flat-class cell by about 2.5×.
