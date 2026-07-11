# Probe suite

A suite of small, capability-isolating synthetic probes used to choose the
default GTLM configuration (bias features, k-hop masking) for the planned
applications: knowledge-graph QA, molecule analysis, relational deep learning,
complex thought trees, and general graph QA. Each probe isolates **one**
capability that one application family needs, so that feature effects are
attributable instead of confounded. Real tasks confirm; probes decide.

The long-range / global-connectivity capability is deliberately **not**
reimplemented here — it is covered by the existing
[`expressiveness`](../expressiveness) experiment (connectivity on large
graphs), whose results are read alongside this suite's.

**Status:** spec agreed 2026-07-03; implemented 2026-07-04; 144-run suite
completed 2026-07-08 (speed/memory results below; accuracy analysis pending
the mask-leak audit). This README is the agreed specification — code must
follow it, deviations go through this file.

Quick start:

```bash
python3 -m sweep src.experiments.probes src/experiments/probes/configs/smoke.jsonc      # local smoke test
python3 -m sweep src.experiments.probes src/experiments/probes/configs/data_prep.jsonc  # build the 4 datasets once
python3 -m sweep src.experiments.probes src/experiments/probes/configs/suite.jsonc      # the 144-run suite
python3 -m sweep.report src/experiments/probes/results/suite                            # aggregate
```

## Suite-wide conventions

| Convention | Value |
|---|---|
| Answer format | Balanced (50/50) **Yes/No**, single-token supervision on the prompt node (as in `expressiveness`) |
| Node labels | Spreadsheet-style `A…Z, AA, AB, …` (`make_node_labels`); exception: `text_path` uses globally-unique person names |
| Bias arms (`bias` config string) | `none`, `spd`, `magnetic`, `spd+magnetic`, `magnetic_shared`, `spd+magnetic_shared` |
| k-hop masking | k ∈ {0, 3} |
| Splits | 4 000 train / 500 val / 1 000 test per probe |
| Reported metric | **Test** accuracy of the checkpoint with the highest **val** accuracy (val evaluated every 100 steps) |
| Seeds | {0, 1, 2}; report mean ± std over seeds |
| Model / recipe | Llama-3.2-1B, LoRA r=8 (α=16, dropout 0.05), lr 1e-5, bias_lr 1e-3, 25 epochs, `v2-flex`, `max-autotune-no-cudagraphs` |
| Also recorded per run | wall-clock s/it, peak GB, token/block sparsity — the speed readout is a first-class result |

Arm semantics:

- `spd` is **directed** SPD — computed on the digraph as-is (what
  `compute_shortest_path_distances` already produces); on undirected probes it
  coincides with symmetric SPD. Symmetric SPD on directed probes was
  deliberately dropped (interpretation control only, re-addable as one arm).
- `magnetic` is the current per-layer implementation; `magnetic_shared` is the
  planned variant computing the bias once per forward and sharing it across
  layers. The arm axis is a free-form string: any new bias variant registered
  in `src/models/bias.py` (`BIAS_TYPES`) becomes available as a token with no
  suite changes.

Full matrix: 4 probes × 6 arms × 2 k × 3 seeds = **144 runs** (probes 1–3 are
minutes-fast; `text_path` is the slow tier).

---

## Probe 1: `direction` — directed reachability

*Capability: using edge direction. Proxy for: KGQA, RDL, thought trees.*

- **Graph:** random connected undirected graph, N ~ U(100, 400); each edge is
  then oriented — one direction only (p ≈ 0.4 each way) or kept bidirectional
  (p ≈ 0.2). Rates are a config knob, tuned so label balance is stable under
  generation. Node text = spreadsheet label (~1–3 tokens); L ≈ 300–1300.
- **Question:** “Is there a directed path from X to Y?” (prompt node linked to
  X and Y, as in `expressiveness`).
- **Labels:** positive = X→Y reachable; negative = not reachable. Because the
  undirected skeleton is always connected, **every** negative pair is
  undirected-connected — symmetric/undirected features provably cannot
  separate the classes; direction is the only signal. The generator records
  the negative mix (reverse-reachable vs. mutually-unreachable) as metadata.
- **Reads on:** directed-SPD vs. magnetic vs. nothing — the decisive
  cheap-vs-expensive directionality comparison.

## Probe 2: `substructure` — ring membership

*Capability: detecting cycles/substructure. Proxy for: molecules.*

- **Graph:** undirected, molecule-like: N ~ U(10, 50), max degree 4, built by
  gluing rings (size 3–8) and attaching trees — matches small-molecule
  statistics (ZINC ≈ 23 nodes, ogbg-mol ≈ 25). Node text = spreadsheet label
  (1 token/node).
- **Question:** “Is node X part of a ring?” Ground truth: X lies on some
  cycle (ring nodes vs. tree nodes, known by construction).
- **Labels:** X sampled 50/50 from ring nodes and tree nodes.
- **Reads on:** where spectral features classically win and pure distance
  features are weak; on this undirected probe `magnetic` degenerates to plain
  Laplacian information — confirming that is itself a result.

## Probe 3: `local_hop` — attribute-in-ball

*Capability: local relational lookup; interaction with k-masking. Proxy for:
RDL, KGQA.*

- **Graph:** same generator as `direction` (oriented connected graph with
  bidirectional-edge rate), N ~ U(100, 300). Node text = `<label>: <color>`
  (~3 tokens), colors from a ~10-word vocabulary.
- **Question:** “Is there a `<color>` node within 2 hops of X (following edge
  directions)?” Radius fixed at r=2 — strictly inside the k=3 mask, so the
  answer is always within the masked receptive field.
- **Labels:** positive = some target-colored node at directed distance ≤ 2
  from X; negative = target color present **only** at distance > 2. Every
  example (positive and negative) contains at least one out-of-ball decoy of
  the target color, so “color exists somewhere” carries no signal and global
  attention is actively distracted.
- **Reads on:** whether k-hop masking is harmless-to-helpful when information
  is local by construction; SPD's home turf.

## Probe 4: `text_path` — people-bios reachability

*Capability: relational reasoning when text dominates (L ≫ N). Proxy for:
complex thought trees.*

- **Nodes:** persons. N ~ U(25, 40), ~40–60 tokens/node, L ≤ ~2.5k. Node text
  = templated sentences, order shuffled except name-first: full name
  (globally unique, gendered name pool; pronouns consistent), 2–3 likes, hair
  color, eye color, age + occupation, city, then relation sentences naming
  each out-neighbor. The attribute sentences serve as *natural* filler — they
  carry no task signal and are sized so bios land in the token budget
  (measured: ~41 tokens/node mean, p95 ≈ 51, L ≤ ~1.7k).

  > “Joe Shmoe likes chess and sailing. He has brown hair and blue eyes. His
  > sibling is Jonathan Shmoe and his colleague is Ann Wu.”

- **Graph:** same construction as `direction`, at bios scale — connected
  undirected skeleton with a degree cap (so texts respect the token budget),
  each edge oriented one way or kept bidirectional (rates tunable for label
  balance). Out-degree ≤ 3; a person's text mentions each out-neighbor, and
  a bidirectional edge is mentioned by both endpoints. Relation words
  (sibling, colleague, friend, …) are pure flavor — no uniqueness or
  matching constraints are needed, because reachability questions never
  resolve individual labeled chains.
- **Text ↔ edges:** fully redundant — every textual mention is exactly one
  graph edge and vice versa. The task is solvable from text alone, so the
  no-bias arm competes fairly: it must *extract* the edge list from prose,
  which is what makes this an integration test as well as a dilution test.
- **Question:** the `direction` question in bios clothing: “Is there a chain
  of acquaintances leading from Joe Shmoe to John Jones?” Positives/negatives
  exactly as in `direction` (all negatives undirected-connected; negative mix
  recorded).
- **Reads on:** task semantics are identical to `direction`, so
  `direction`-accuracy minus `text_path`-accuracy, per arm, is a controlled
  measurement of text dilution; plus the per-arm wall-clock overhead at small
  N / large L (expected ≈ free — this probe verifies it).
- **v2 (deferred):** a nested-chain variant (“Does the best friend of Joe's
  sibling have green eyes?”) with per-type matching constraints and in-graph
  decoys was fully specified and then deliberately deferred — it adds a
  labeled-chain capability readout but with a confounded (text-skill vs.
  bias-channel) interpretation and the suite's most complex generator. If it
  is ever built, it is one more task-registry entry; it may belong to the
  thought-tree application experiment instead.

---

## Decision rule (agreed 2026-07-04)

One rule over the full config grid (6 arms × 2 k = 12 configs), with
ε = **2 pp** (accuracy is always the mean over seeds {0, 1, 2}):

1. Per probe *p*, let `best(p)` = the highest accuracy over the 12 configs.
2. **Candidate set** `C` = configs within ε of `best(p)` on **every** probe.
   (No-regression is built in: a config that wins on two probes but tanks on
   another is out.)
3. **Default** = the cheapest config in `C`, by wall-clock s/it measured on
   `direction` (the suite's worst case for bias cost).
4. The magnetic and k-masking conclusions are reported as **commentary** on
   which configs made it into `C` — they are not separate binding rules.

Provisos:

- **Long-range (expressiveness) results are advisory only.** The existing
  big_test numbers (single seed, 200-example eval, no `magnetic_shared` arm)
  inform the discussion but do not enter the quantifier in step 2.
- **Fallback — no global winner:** if `C` is empty, defaults go
  per-application: map probes to application families
  (`direction`/`local_hop` → KGQA + RDL, `substructure` → molecules,
  `text_path` → thought trees), rebuild `C` per family over that family's
  probes, and pick the cheapest candidate per family. The global default is
  deferred.
- **Cost sanity check:** if the chosen default's s/it exceeds ~2× the
  cheapest config's, the result is flagged as a genuine accuracy-vs-cost
  trade-off for explicit sign-off rather than auto-adopted.
- The suite is a permanent regression harness: any future bias variant is
  evaluated by adding its arm token and re-running the sweep, not by ad-hoc
  experiments.

---

## Results — speed & memory (144-run suite, 2026-07-08)

Wall-clock **s/it** (`train_runtime_s / n_steps`) and **peak CUDA GB**, mean
over seeds {0, 1, 2}, from `results/suite/runs.jsonl` (B300, bf16, `v2-flex`,
`max-autotune-no-cudagraphs`). Accuracy readouts are reported separately —
the k=4 accuracy columns of the reachability probes are under audit for a
mask→label leak and should not be quoted from this sweep.

| bias / k | direction s/it · GB | local_hop s/it · GB | substructure s/it · GB | text_path s/it · GB |
|---|---|---|---|---|
| none k0 | 1.09 · 11.8 | 1.15 · 20.8 | 0.96 · 11.7 | 1.73 · 38.7 |
| none k4 | 1.17 · 11.8 | 1.24 · 20.8 | 0.96 · 11.7 | 1.84 · 38.7 |
| spd k0 | 1.38 · 12.9 | 1.77 · 21.9 | 1.07 · 11.7 | 3.10 · 38.7 |
| spd k4 | 1.40 · 12.9 | 1.79 · 21.9 | 1.14 · 11.7 | 2.37 · 38.7 |
| magnetic k0 | 6.34 · 13.1 | 5.52 · 21.9 | 1.47 · 11.7 | 3.26 · 38.7 |
| magnetic k4 | 6.33 · 13.1 | 5.49 · 21.9 | 1.33 · 11.7 | 2.53 · 38.7 |
| magnetic_shared k0 | 1.38 · 15.4 | 1.70 · 23.5 | 0.98 · 11.7 | 3.10 · 38.7 |
| magnetic_shared k4 | 1.35 · 15.4 | 1.68 · 23.5 | 0.93 · 11.7 | 2.41 · 38.7 |
| spd+magnetic k0 | 6.70 · 13.1 | 5.83 · 21.9 | 1.57 · 11.7 | 3.23 · 38.7 |
| spd+magnetic k4 | 6.77 · 13.2 | 5.84 · 21.9 | 1.53 · 11.7 | 2.47 · 38.7 |
| spd+magnetic_shared k0 | 1.63 · 16.4 | 2.01 · 24.6 | 1.07 · 11.7 | 3.15 · 38.7 |
| spd+magnetic_shared k4 | 1.74 · 16.4 | 2.00 · 24.6 | 1.12 · 11.7 | 2.35 · 38.7 |

Takeaways (speed first — it is the axis that actually varies):

- **Per-layer `magnetic` is the only expensive arm, and only at large N:**
  5.5–6.8 s/it on the 100–400-node probes — 4–6× the no-bias baseline — but
  near-free on substructure (10–50 nodes). The cost is the per-layer
  recomputation, not the eigendecomposition: **`magnetic_shared` runs at SPD
  cost** (1.4–2.0 s/it) everywhere, confirming the shared implementation
  achieves its design goal.
- **The probe-4 "≈ free" expectation is refuted.** At small-N / large-L
  (`text_path`), *every* bias arm — including spd and shared — costs ~1.8×
  baseline (3.1–3.3 vs 1.7 s/it at k=0). The overhead is uniform across arms,
  so it is the token-level application of the (L, L) bias per layer, not the
  bias computation. This is a fixed tax on the thought-trees regime,
  independent of which bias is chosen.
- **k-masking buys little throughput at these scales:** ~0 on the
  spreadsheet-label probes (block-sparsity gains offset by mask overhead) and
  a modest ~25% on `text_path` (2.4 vs 3.1 s/it).
- **Memory is set by the task, not the config:** ~11.7 GB (substructure) to a
  flat 38.7 GB (`text_path`, L ≈ 2.5k) across all twelve arms; k has zero
  effect. One caveat: `magnetic_shared` is the most memory-hungry *family*
  (+3.6 GB over baseline on `direction`, vs +1.3 for per-layer) — the shared
  (N, N) bias lives outside gradient checkpointing, so it is the term that
  grows as N² if application graphs outgrow ~400 nodes.
