# The graph generalist — every graph domain, one model, released

**Status (2026-10-03):** planned, nothing built past the molecule generalist. This is the plan of
record for taking the generalist from molecules to every graph domain in the repo and releasing the
result. `PLAN.md` holds the programme-wide decisions (D1–D7, held-out set, forgetting control) and
`DESIGN.md` the harness; this document says what the multi-domain campaign does with them, and in
what order. `MOLECULE_GENERALIST.md` is the campaign this one extends: its trunk, anneal-with-replay
recipe and assistant pipeline are the starting point for everything below.

**What gets released, and what each thing is for:**

| artifact | what it is | used how |
|---|---|---|
| **trunk** (1B, 8B; 3B conditional, §8) | GTLM pretrained on every in-mixture graph task (§2): a text+graph model, not assistant-tuned | the starting point for specialist fine-tuning, in place of training from base Llama |
| **assistant** (same sizes) | the trunk annealed on the assistant set (§6) with text replay in the decay | zero- and few-shot graph-conditioned QA, no further tuning |
| **graph assistant dataset** | the §6 set, every answer composed from computed facts, with a dataset card | open data for graph-conditioned instruction tuning |
| **inference package** | modeling code, graph input format and collator, an example (§9) | what makes the checkpoints usable by anyone else |

Everything ships on the Hugging Face Hub.

---

## 1. Stages

1. **Graph pretraining — the trunk.** One mixture over every in-mixture source of §2, one shared
   recipe (§4), WSD with no decay. This is `PLAN.md` Phase 1 and §5, run at 1B, with the molecule
   mixture entering as one block.
2. **Assistant data.** The `assistant/` pipeline made task-agnostic (§6), fed by the trunk's own
   graphs plus disguised synthetic graphs.
3. **Assistant anneal.** A fork of the trunk, decayed on trunk mixture + assistant set + text replay —
   the recipe `MOLECULE_GENERALIST.md` §2 measured (replay in the decay is close to free, in the
   mixture it is not).
4. **Trainable surface.** LoRA unless the written trigger of §5 fires; checked at 1B, before any 8B
   run.
5. **Scale** to 8B (§8), recipe re-bracketed, not re-designed.
6. **Release** (§9), with the licence constraints of §10.

Multi-turn data is deferred until single-turn grounding holds on the off-generator set (§7). When it
comes, the first form is a clarify-then-answer second turn: the pipeline already renders the
clarifying question (`render.can_clarify`), so the second turn stays verifiable.

---

## 2. What goes in

| source | in the mixture | kind | notes |
|---|---|---|---|
| `graphqa` | the reported tasks minus the held-out pair; `disconnected_nodes`, `node_classification` train-only | corpus | `baharef/GraphQA` |
| `probes` | `substructure`, `local_hop`, `text_path` | generator | |
| `expressiveness` | HARD, sizes log-uniform 10–1,000 nodes | generator | the widest size range in the mixture; the specialist's 1,600–2,400 ran an 80 GB card out of memory on the dense pair bias |
| `our_tests` | `kg_qa` (synthetic KG-QA) | generator | Family Tree is held out |
| `kgqa` | WebQSP, CWQ — Levi, never triplet | corpus | |
| `tag_benchmarks` | cora, ogbn-arxiv, reddit | corpus | Pubmed held out; the test-selection re-run is owed before TAG numbers anchor anything |
| molecules | the `008` mixture as one block: Tier A, BACE/BBBP/HIV, Tox21/SIDER, ChEBI-20, graph-to-SMILES | both | as in `MOLECULE_GENERALIST.md` §1 |
| text replay | `text/replay` | generator | decay only (§1 stage 3) |

**Out:** `context` (saturated; eval-only for length extrapolation), `relbench` (graph arm −7.7 pp on
rel-trial with the bias channel inert, and the worst corner of the cost surface), and
`graphqa_mag_khop` (an ablation over GraphQA's data, not a source). `permutation_equivariance` and
`backward_compatibility` are tests, not data.

**Held out from all training**, as declared in `PLAN.md` §3.3 before any run: GraphQA triangle
counting and connected nodes, the `direction` probe, Family Tree, Pubmed, ClinTox, `bond_path`,
`longest_chain`. The assistant set inherits the same exclusions (§6).

Every source enters through an adapter (`DESIGN.md` §D3: `build / load / partition`); only
`molecules` and `text` exist. Corpora keep their pass caps; generators refresh per pass.

---

## 3. Batching — by shape, synchronised across ranks

Mixing tiny molecules with 2,000-node graphs makes shape variety the throughput problem: every new
`(B, L, N)` triple is a Triton autotune, and the molecule anneal lost 96 % of its wall clock to them
(`DESIGN.md` §D9). Two changes, both in `mixture.py`:

* **Rank-synchronised micro-steps keyed by shape.** Today rank *r* takes micro-batches
  `[r::world_size]` of a step, so ranks run different buckets side by side and wait on the slowest.
  Instead, a step's micro-batches are emitted in groups of `world_size` drawn from **one** bucket
  with **one** row count, so at micro-step *j* every rank does the same work on the same compiled
  shape. The key is shape, not task family: a family is a poor proxy for size (CWQ subgraphs and
  expressiveness graphs span wide ranges, GraphQA and the probes overlap), and one family per step
  would hit the shared bias parameters in one lump every ~50 steps for a family at 2 % weight. The
  optimizer step stays mixed. *Built 2026-10-05.* The bucket key is the collator's own padded shape
  (`wiring.collator_shape_fn`), the sampler reshapes the step to exactly `accumulation_steps` groups
  itself (a per-rank reshape could pick different merges on different ranks), and each micro-batch
  carries its group's shape to `wrap_collator`, which pads to it — a merged group holds rows of
  several sizes, and padding each rank's batch to its own widest row would desynchronise them again.
  When a step's total is not a multiple of the rank count, one group per step is a row short on
  some ranks; the shape still matches.
* **A fitted bucket ladder.** The fixed power-of-two ladder (floors 8 nodes, 32 tokens) is replaced by
  boundaries fitted to the measured `(N, L)` distribution of the whole mixture: minimum padding for
  a given bucket count, with the shape set — `B` included — well inside the compile cache
  (`wiring.FLEX_CACHE_SIZE_LIMIT`, 512), since evaluation shares it. One measurement pass over the built data.
* **A row cap that counts what the collator builds.** `batches_for_step` caps a micro-batch at
  `micro_batch_tokens // token bucket` on the 32-token ladder, but the flex collator pads every row
  to a multiple of 512 tokens and a power of two of nodes (floor 32). Molecule rows (~200+ tokens)
  hid the gap at about 2x. The graph domains do not: a 33-token GraphQA row admits 64 rows a
  4,096-token micro-batch, which pad to 32,768 positions, and the logits alone ran an 80 GB card out
  of memory in the 2026-10-04 smoke. The cap has to be computed on the padded `(L, N)`; the dense pair
  bias makes N² the second budget beside B·L. *Built 2026-10-05, and a cap alone was not the fix:*
  `tokens_per_step` fixes a step's example count from raw lengths and HF fixes how many micro-batches
  a step is cut into, so with a configured accumulation a short-row step still lands on the card at
  many times its raw tokens however it is grouped. `micro_batch_tokens` (padded tokens per micro-batch
  per rank) and `micro_batch_node_pairs` (rows × N²) are now memory budgets, and the accumulation is
  derived from them: the most groups any of the next 500 steps is cut into
  (`MixtureSampler.derive_accumulation_steps`), so a step is only ever split and never merged past
  the budget. The first version divided the mean padded volume by the budget. That is a lower bound,
  because groups of different shapes do not pack, and on 2 ranks it gave 2 micro-batches where the
  bucketing made more; the merged groups ran an 80 GB card out of memory. The accumulation is now
  long on the 512-step ladder, about one micro-batch per shape bucket: 13 a step on the smoke
  mixture at 1 rank, 12 a rank at 2. That costs nothing measurable. The 60-step H100 smoke ran in
  2,868 s at a 21.8 GB peak on 1 rank, against 4,684 s and 50.9 GB with the volume rule's 4, and in
  2,129 s at 24.1 GB on 2 ranks. Unset, a run keeps its configured accumulation.

**The draw distribution must not move.** Per-task counts are drawn before batching, from the
mixture weights, and batching only regroups them. The one way shape-keying could bias the draw is a
bucket too small to give every rank a batch at a micro-step: those examples are **promoted to the
next larger bucket** — paying padding — and never dropped or deferred, since deferring would
under-sample whatever is unusually sized. A test pins it: a step's per-task counts are identical
before and after batching, at 1 and at 4 ranks. The `grad_share` readout checks the same thing in
training.

**A run never silently runs out of data.** A corpus past its `passes` cap used to retire with an
info line and a generator asked for an unbuilt pass hours in; the molecule forks lost runs to both.
`MixtureSampler.check_supply` now replays the draw plan to the end of the job before the first step
— the per-step counts are a pure function of the step, so the replay is exact — and refuses either.
A smoke run whose 64-row sources are meant to run dry sets `allow_exhaustion`.

D5's caps (`max_nodes`, `max_edges`, `max_tokens`, sampler per oversized task) are still owed and
are chosen against measured s/it on the largest components (expressiveness, CWQ, TAG).

*Measured 2026-10-06* on the full build (config 015), train splits. Two tools, both in `tools/reports/`: `shapes.py` reads each split's recorded node and
token counts and fits ladders; `step_cost.py` times one micro-batch per padded shape (the smoke's
1B model and LoRA, `micro_batch_tokens` 8,192, forward + the trainer's loss + backward, on a B200).

* **Step cost follows padded tokens, plus an N² term from N = 512 up.** A full 8,192-token
  micro-batch takes ~230 ms at N ≤ 128, 330 ms at N = 256, 630 ms at N = 512 and ~1 s at N = 1,024.
  Steady-state ms per row: GraphQA 15, WebQSP 25, cora 32, expressiveness 41, arxiv 56, text_path
  76, CWQ 106. Peak memory stays under 38 GB at this budget. Reddit was not built when this ran;
  its rows (32 nodes, ~1,000 tokens) sit on shapes cora and arxiv already cover.
* **CWQ is the oversized task.** Its rows above 3,072 tokens are 20 % of the rows and about
  70 % of its cost; the 1.8 % above 4,608 cost 647 ms a row. Its token p99 is 4,518, but the
  longest row is 11,149 tokens. A `max_tokens` cap of 4,608 drops 0.85 % of CWQ rows and leaves
  every other source untouched.
* **Every new shape costs 20–140 s of compile and autotune** on first touch, per rank, unless
  the inductor cache is primed: about 14 minutes for the 20 shapes these sources populate.
* **The ladder.** With every domain weighted equally and the 4,608 cap, the current ladders pad
  tokens 1.67× and node pairs 1.74× over 32 populated `(N, L)` shapes. A fitted 6 × 6 ladder
  (tokens 128/384/768/1,280/1,792/4,608; nodes 64/176/304/512/736/1,040) pads 1.38× and 1.45× over
  28 shapes: better on both counts. At 8 × 8 it pads 1.28× and 1.32× over 43 shapes. GraphQA is
  most of the gap: its 33–98-token rows pad to 512 on the current ladder.

Pending a decision: the CWQ cap (4,608, or lower with a per-task subgraph sampler), the ladder size
(6 × 6 or 8 × 8), and whether the budget moves now that 8,192 tokens uses under half an 80 GB card.

---

## 4. One recipe

Every per-task campaign tuned its own recipe to win as a specialist; the trunk cannot have eight. One
shared recipe, chosen once:

* **Screen** at 1B: `lr ∈ {5e-5, 1e-4}` × LoRA `r ∈ {16, 64}`, one seed each, short budget,
  `bias_lr` at its default 5e-3. Selection is the mean over in-mixture tasks of each task's score
  normalised to its specialist. Spare GPUs go to untried cells, not seed replicates.
* **Kept fixed:** `lora_dropout 0.15`, `question_node: isolated`, `spd+magnetic` at `G=4`
  (`PLAN.md` D2), `max_spd 32`, `tokens_per_step 16384`, chat formatting on Instruct weights.
* **The `bias_lr` : `lr` ratio stays constant** through every phase, as it has in every run so far.
  A schedule that decays it toward 1 was considered and not adopted: the distinct-rate setup works,
  and the ratio is partly a permanent scale correction between parameter groups, not only a
  cold-start effect.

The winning cell is written here with its numbers before the trunk starts.

---

## 5. Trainable surface — LoRA, full fine-tuning as the last resort

**LoRA is the default.** It reached specialist level on molecules, it is cheap at 8B, and switching
the adapter off gives the text-replay KL a free reference model. The `W_q`/`W_k`-unfrozen variant
(`PLAN.md` D4 arm B) is dropped: it adds a third arm and its own optimizer-group plumbing for an
effect unlikely to be large.

**Full fine-tuning runs only if this trigger fires, at 1B, before any 8B run:**

1. the 1B trunk falls short of the per-task specialists by more than seed noise on a majority of
   in-mixture tasks, **and**
2. raising the LoRA rank (one step past the §4 winner) does not close the gap.

If it fires, full fine-tuning is run at 1B against the LoRA trunk on the same mixture and budget.
At 8B it needs B300s or ZeRO-2 (`PLAN.md` D4), and the text-replay reference becomes a frozen copy
or precomputed logits.

---

## 6. Assistant data — one task-agnostic pipeline

The pipeline already separates what is shared from what a domain supplies: `Fact` and the intent →
render → write → judge → accept → compose chain know nothing about molecules, and `Domain` holds the
vocabulary. The target is that **a new task costs a fact extractor and a template, nothing else.**
What has to change in the core for that to hold:

| limitation | fix |
|---|---|
| answer kinds are closed — `count / yesno / smiles / text` | add `entity`, `entity_set`, `sequence` (paths, orderings) and `label`, each with a renderer and an acceptance check |
| the unsupported-claim check is molecule vocabulary (ring systems, compound classes) | a generic rule: every entity name and number in a reply traces to the sheet or the question; a domain may add stricter checks |
| a benchmark carries about one fact per graph — its label | a **shared structural extractor** over any graph (degree, neighbours, paths, cycles, connectivity, counts — networkx), plus a per-task label extractor |
| polarity: `fg_presence` shipped at a 0 % yes-rate, the largest molecule defect | part of the contract — every yes/no family emits matched negatives, and the draw balances polarity per family |
| fact sheets for large graphs (CWQ, expressiveness, TAG nodes with long text) overflow the writer and swamp the checks | a bounded sheet sampled around a focus region of the graph |
| a closed unanswerable vocabulary — the decline did not survive new phrasings | each template declares plausible-but-absent fact families; other domains' families add more |
| part references differ by task — atom indices, integer ids, entity names, paper titles | the template supplies the reference pattern and noun; the shared code already reads them from `Domain` |

The per-task template is a `Domain` subclass: reference pattern, nouns, ask phrases, glosses,
absent families, and `example_builder`.

**Where the graphs come from.**

* **The trunk's own graphs.** Assistant rows inherit the trunk's partition and held-out exclusions,
  so no new leakage path opens, and the assistant skill is separated from graph novelty.
* **Disguised synthetic graphs**: an abstract graph and its computed facts, presented as a
  real-world situation (a transit map, an org chart, a supply chain). The disguise is a mapping
  owned by Python — node → entity name, edge → relation — drawn from a closed scenario table with
  fictional names, and the writer only voices it. A writer that invents the disguise adds
  world-knowledge claims the graph does not support. The shared structural extractor works on these
  graphs unchanged.

**Order of domains**, by how cheaply their facts are verified: generic graphs (GraphQA, probes,
expressiveness, synthetic) → knowledge graphs (WebQSP, CWQ, `kg_qa`) → text-attributed graphs →
molecules (exists; ported onto the generalised core).

The writer is `gemma-4-31B-it` from the real `-it` snapshot, with `tools/checks/chat_template.py` run
before any build. A hand read of every new domain's first build is part of the build, not a review
after it: on molecules it found 16–22 % defects past every automatic check.

---

## 7. Evaluation

Every set below is built and frozen before the run it scores.

* **Interference** — each in-mixture task on the annealed trunk against its specialist number.
  These exist for every task in §2.
* **Transfer** — the held-out tasks, zero-shot and as adaptation efficiency (steps to target from
  the trunk vs from base), on the k-fold harness (`KFOLD_TRANSFER.md`). Property prediction is the
  caveat that study found: the trunk arrives sooner but ends below scratch. The release claim for
  the trunk is stated per task family, not as one number.
* **Assistant, on-generator** — the held-out split of the assistant set.
* **Assistant, off-generator** — hand-written questions per domain plus disguised graphs from
  scenarios absent from training. This is the set that decides whether zero-shot use is claimed;
  on molecules the model was right on 8 of 29 facts there while scoring well on its own split.
* **Text behaviour** — `text_behaviour` and the general-knowledge caption check, per anneal.

---

## 8. Scale

| size | backbone | layers | role |
|---|---|---:|---|
| 1B | Llama-3.2-1B-Instruct | 16 | every decision in §3–§6 is made here |
| 3B | Llama-3.2-3B-Instruct | 28 | conditional: run only if the 1B→8B gap makes a middle point worth its cost |
| 8B | Llama-3.1-8B-Instruct | 32 | the headline release |

Moving up re-brackets `lr` (the KGQA 8B optimum was 5e-5 on both WebQSP and CWQ, on base weights)
and re-profiles D2's `G` at 32 layers; it changes nothing else. Long 8B runs need ≥128 GB host RAM
per rank (`kgqa` README, base-model scale). Budget: `PLAN.md` §8, ~3–5k GPU-h for a first 8B cycle.

---

## 9. Release

* **Names start with "Llama"** and every card and page carries "Built with Llama" — the Llama 3.1 /
  3.2 licence requires both for any model trained from Llama materials.
* **Modeling code** ships as `trust_remote_code` files pinned to `transformers` 4.50.3, the version
  the GTLM attention internals are written against.
* **Graph input** ships as a small installable package: the graph format, the collator and an
  end-to-end inference example. Generation is HF `generate()` only; vLLM/SGLang have no path for
  the attention biases.
* **Model cards** state: built on Instruct weights with chat formatting; the trunk has trained on
  the train splits of every §2 benchmark; the held-out tasks; the Reddit-derived training data
  (§10).
* **Dataset card** states how every answer is composed (computed facts, writer voices only), the
  hand-read defect rate, the writer model, and the licence per subset.

---

## 10. Licences

| source | licence | consequence |
|---|---|---|
| Llama-3.2-1B/3B-Instruct, Llama-3.1-8B-Instruct | Llama community licence | naming and attribution of §9; acceptable-use policy applies |
| `gemma-4-31B-it` (writer) | Apache-2.0 | no restriction on the generated set |
| GraphQA (`baharef/GraphQA`) | CC-BY-4.0 | attribution |
| WebQSP (`KGQA/KGQA-datasets`), CWQ (`drt/complex_web_questions`) | Apache-2.0 | attribution |
| TAG reddit (RGLM → GLBench, from SNAP) | non-commercial use | Reddit-derived assistant rows ship as a separate subset under non-commercial terms, so the rest of the set keeps permissive licences |
| cora, ogbn-arxiv (RGLM → LLaGA) | MIT as repackaged; ogbn-arxiv ODC-BY upstream | attribution |
| ChEBI-20, MoleculeNet sets | ChEBI CC-BY-4.0; MoleculeNet per set | confirm per set before the dataset release |
| probes, expressiveness, `our_tests`, synthetic graphs | generated here | ours |

---

## 11. Build order

- [x] adapters: graphqa, probes, expressiveness, `our_tests/kg_qa`, kgqa, tag (§2)
- [ ] D5 caps and per-task samplers, chosen on measured s/it (§3)
- [ ] shape-keyed rank-synchronised batching, fitted ladder, draw-invariance test (§3)
- [ ] plumbing smoke on three maximally different tasks across domains (`PLAN.md` §10)
- [ ] recipe screen at 1B (§4) → recipe written here
- [ ] 1B trunk; interference and transfer read (§7); trainable-surface trigger evaluated (§5)
- [ ] assistant core generalised (§6) and molecules ported onto it with no regression on its set
- [ ] off-generator evaluation sets built and frozen per domain (§7)
- [ ] assistant domains in §6 order, each hand-read on its first build
- [ ] 1B assistant anneal; on- and off-generator read
- [ ] inference package and cards (§9), checked from a clean environment
- [ ] 8B: `lr` re-bracket, D2 re-profile, trunk, anneal; 3B if §8's condition holds
- [ ] Hub release
