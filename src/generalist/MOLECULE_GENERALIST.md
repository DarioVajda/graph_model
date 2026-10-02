# The molecule generalist — one model over every molecule task

**Status (2026-09-20):** campaign run and read out. Trunk of record
`configs/probes/008_molecule_generalist_instruct.jsonc` — six annealed 1B checkpoints, 3 seeds × 2
arms, `Llama-3.2-1B-Instruct`. §4 is part-run.

**The question.** Does a graph channel earn its place at 1B when one model does *every* molecule task
at once? Two arms, identical data, one difference:

| arm | sees | graph-to-SMILES is |
|---|---|---|
| **graph** | `rich_levi` graph, stereo tags on, no SMILES anywhere in the prompt | a real task |
| **flat twin** | the SMILES string, matched form of every task | canonicalizing a string that already spells the answer |

Same molecules, mixture, budget and seeds, so any gap is the representation. `DESIGN.md` is the *how*.
Full working record — all tables, three earlier campaigns, the assistant defect log — at `dece0c7`;
section numbers cited in code comments (`§5`, `§7.6`, `§9.4`) are that version's.

---

## 1. What goes in

| source | tier | form | metric | role |
|---|---|---|---|---|
| 9 RDKit families (rings, functional groups, stereo) | A | 1–3 token answer | exact match | train + test |
| BACE, BBBP, HIV | B | yes/no | ROC-AUC from the margin | **headline** |
| Tox21, SIDER | B | yes/no per endpoint | ROC-AUC | training signal only — no anchor ladder |
| ChEBI-20 | C | free-text caption | BLEU/ROUGE/METEOR | in-mixture diagnostic |
| graph-to-SMILES | — | canonical SMILES, stereo-free | validity, round-trip, exact | train + test |
| `bond_path`, `longest_chain`, ClinTox | — | — | — | **held out**, no training source touches them |

One molecule-level partition spans every source: a molecule is train or test *everywhere*. `lr 1e-4`
matched across arms, WSD, one anneal to `lr/10`; **the annealed checkpoint is the reportable model**,
no test-set or best-val selection. Arms matched in *examples*, not tokens (288 vs 83 tokens/example).
~80 GPU-h.

## 2. Results

**Graph-to-SMILES — the widest gap in the campaign.**

| | graph | flat |
|---|---:|---:|
| `exact_match` | **0.4193** ±0.0291 | 0.2113 ±0.0031 |
| `roundtrip_match` | **0.4613** ±0.0261 | 0.2607 ±0.0101 |
| `validity` | **0.7560** ±0.0156 | 0.6607 ±0.0117 |

Graph leads in every heavy-atom bucket (0.54/0.50/0.52/0.49/0.22 against 0.44/0.28/0.28/0.20/0.09),
degrading only past 31 atoms where the node budget bites.

**Property prediction — a null.**

| task | graph | flat | gap |
|---|---|---|---:|
| BACE | 0.8303 ±0.0390 | 0.8599 ±0.0208 | −0.0296 |
| BBBP | 0.7061 ±0.0237 | 0.6973 ±0.0333 | +0.0088 |
| HIV | 0.7618 ±0.0373 | 0.7575 ±0.0449 | +0.0042 |
| SIDER | 0.8586 ±0.0044 | 0.8449 ±0.0037 | +0.0138 |
| Tox21 | 0.8373 ±0.0136 | 0.8231 ±0.0078 | +0.0142 |
| **5-set mean** | **0.7988** | **0.7966** | paired −0.0022, sd 0.0126, t(2) = −0.31 |

Not significant (t crit. at df=2 is 4.303). On base weights the flat arm led +0.0095; the swap flipped
the sign and narrowed the spread. Notation ladder: the flat edge is *SMILES*-specific — InChI −0.0138
and SELFIES −0.0167 against SMILES +0.0135, same molecules and mixture.

**Structural probes — exact match. Every gap keeps its sign from base weights except `stereo_potential`.**

| task | graph | flat | gap |
|---|---|---|---:|
| `ring_size` | 0.9667 ±0.0122 | 0.8893 ±0.0266 | **+0.0773** |
| `fg_atom_membership` | 0.9973 ±0.0031 | 0.9760 ±0.0122 | +0.0213 |
| `ring_membership` | 1.0000 ±0.0000 | 0.9940 ±0.0035 | +0.0060 |
| `fg_presence` | 0.9933 ±0.0046 | 0.9873 ±0.0012 | +0.0060 |
| `fg_count` | 0.9640 ±0.0020 | 0.9700 ±0.0087 | −0.0060 |
| `stereo_assigned` | 0.9860 ±0.0040 | 0.9973 ±0.0012 | −0.0113 |
| `ring_count` | 0.8993 ±0.0101 | 0.9380 ±0.0122 | −0.0387 |
| `longest_chain` *(held out)* | 0.0780 ±0.0035 | 0.0240 ±0.0151 | **+0.0540** (3.3×) |
| `bond_path` *(held out)* | 0.0760 ±0.0040 | 0.0473 ±0.0023 | +0.0287 (1.6×) |

Held-out pair is at floor accuracy — the ordering is the finding, not the score.

**Captioning — the one family where flat leads consistently.** A caption is text about a molecule, and
the flat arm's molecule is already text.

| | graph | flat | gap |
|---|---|---|---:|
| `bleu2` | 0.4257 ±0.0021 | 0.4567 ±0.0095 | −0.0309 |
| `rouge_l` | 0.5050 ±0.0018 | 0.5334 ±0.0077 | −0.0285 |

In-mixture diagnostic, **not** a publishable ChEBI-20 row: measured the publishable way the same
checkpoints give graph 0.3290 ±0.0028 / flat 0.3533 ±0.0044 BLEU-2.

**Invariance and leakage.** Property 1 holds 15/15 with permutation spread at or below its own control
(flat ran 10–40× wider on base weights). Property 2: every backbone weight exactly base Llama's on all
six cells. Leakage passes on all six.

**The cost, and the fix.** Asked 48 general-knowledge questions with no molecule in them, the graph arm
answers one in seven with a ChEBI-20 caption. `text/replay` — single-node graphs whose targets are the
backbone's own continuations — removes it entirely; *where* it is spent decides the price:

| replay | caption rate | property mean Δ | g2s Δ |
|---|---:|---:|---:|
| none (trunk) | 0.139 ±0.012 | — | — |
| 15 % of the **mixture** | **0.000** | −0.0170 (−8.2 sd) | −0.074 |
| 8 % of the **mixture** | **0.000** | +0.0052 | −0.036 |
| 15 % of the **decay** | **0.000** | **−0.0024 ±0.0076** | **−0.008** |
| 40 % of the **decay** | **0.000** | −0.0016 ±0.0056 | −0.029 |

Replay in the decay is close to free; in the mixture it is not. g2s is the only monotone casualty.

> **Caveats.** Three seeds throughout; the doubled horizon tripled the paired sd (0.0071 → 0.0201), so
> anything built on the arm comparison needs more. Tox21/SIDER are not anchor-comparable. Generation
> numbers from the three earlier campaigns are void — measured through a stop-token defect that scored
> whether the model stopped, not whether it was right.

## 3. Where it stands

The annealed graph checkpoints are a **trunk**; what remains measures them as a starting point.

| | state |
|---|---|
| **Replay in the trunk** | **done.** `008`'s graph cells stay the trunk; reportable anneal becomes `replay_anneal15` — one extra anneal/seed (~1 GPU-h) vs 65 GPU-h for a new trunk |
| **In-mixture specialisation** — fork reaches the specialist's score in what fraction of its steps | not run |
| **Held-out adaptation** — `bond_path`, `longest_chain`, ClinTox | not run; owed since the harness was designed |
| **The assistant** — free-form graph-conditioned questions | **data built, not wired** |

**Assistant set:** 10,640 examples (9,873 train / 767 test). Every answer composed by Python from an
RDKit fact sheet and only re-voiced by a writer, so invented chemistry is unreachable rather than
filtered. 100 rows read by hand found three defect classes, all counted exactly and removed. Pipeline
is `assistant/pipeline/run.sh`; operating hazards are in the module docstrings.

**Blocking the fork:** `mol/assistant` is not in the registry. `build_assistant_example` and the
add-a-task fork mechanism exist and are tested; missing is the TaskSpec naming where the JSONL lives
and its cost per example.

Also owed: the `bias: none` control, the g2s-only ceiling, the `val` role shrink at the next rebuild.
