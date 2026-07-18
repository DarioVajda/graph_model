# TODO — closing the flat-vs-graph gap (position encoding + capacity)

Status: **COMPLETE** (written 2026-07-16, all 15 runs finished same day —
E1/E2 launched ~15:07, E3 ~15:13, last run landed ~19:45). Every arm reused an
existing result as its control wherever one already existed on record — see
each experiment's "Run" section for the exact source — rather than re-running
settings already trusted. New arms all got 3 seeds. Follow-up to
the flat-beats-graph discussion in the README ("Flat-beats-graph … framed as
regime-specific") and [[kgqa-direction]].

## Overall verdict

**All three probes came back negative** on their respective hypotheses — E1
falsifies the premise E3 was built on, E2 rules out capacity, and E3's fix
itself is a clear regression. Net effect: the flat/graph gap (0.7351 vs
0.7490 test F1, question-node arm vs. flat) is **not** explained by (a) a
retrieval-order → RoPE-position signal flat is exploiting and graph lacks, or
(b) the magnetic bias MLP being too narrow. Read together with `TODO_reg.md`
(the bias channel already carries real, generalizing signal — regularizing it
hurts test F1 more than train-fit) and E3's own Hits@1-preserved-but-F1-collapsed
signature, the honest state of play is: **RoPE-visible relative position was
not the missing ingredient**, and the "impedance mismatch" framing from this
doc's original Motivation section is *narrowed*, not confirmed — it's still
plausible the graph channel is undersized/underdriven relative to what it's
replacing, but "give it a RoPE-shaped distance signal" was the wrong lever.
Whatever closes the remaining ~1.4 F1 gap needs a new hypothesis; see each
experiment's Result section below for the specific evidence and reasoning.
`node_position_mode="spd_depth"` and `flat_shuffle_lines` stay in the
codebase (real, tested, reversible) as documented negative results, not
recommended arms — defaults (`"reset"` / `False`) are unchanged.

## Motivation (see README for the full mechanism writeup)

The question-node arm (`029_question_node_webqsp.jsonc`) closed ~⅓–½ of the
flat/graph gap by fixing question-agnostic graph encoding, but flat still
leads (0.7351 vs 0.7490 test F1). Code-level analysis of the current stack
(`src/models/{bias,structural_mask,attention,causal_lm,context,dispatch}.py`,
`src/utils/text_graph_collator_v2.py`) surfaced three candidate mechanisms,
none previously tested in isolation:

1. **RoPE position reset.** `GraphCollatorV2._pack_one` resets `position_ids`
   to `arange(len)` at every node boundary. RoPE's attention bias is a
   function of `pos_q − pos_k`; resetting every node to 0 means **zero**
   inter-node relative-position signal survives — every bit of "how far
   apart are these two things" must come from the SPD/magnetic bias MLPs.
   Flat gets a real distance cue for free: its triples are laid out in
   retrieval order (correlated with hop-distance from the topic entity), so
   RoPE — the backbone's most heavily pretrained circuit — carries it
   natively.
2. **Impedance mismatch.** The frozen backbone's attention weights were
   pretrained exclusively under causal + RoPE. Graph mode asks them to do
   bidirectional, position-flat attention, compensated only by a
   freshly-initialized (bar zero-init output layers) bias channel trained
   for ~15 epochs at LoRA scale — vs. RoPE's billions of pretraining tokens.
   `TODO_reg.md` already showed the bias channel learns *real*, generalizing
   structure (regularizing it hurts test F1 more than train-fit) — so this
   isn't a "the bias channel is broken" story, it's a "the bias channel is
   undersized/underdriven relative to what it's replacing" story.
3. **Duplicate pre-attention keys.** Levi relation-nodes aren't deduplicated
   (many triples share identical relation text); combined with (1), two such
   nodes are bit-identical in Q/K/V at layer 0. The pairwise bias can still
   separate their attention *scores*, so this mostly self-resolves by layer
   1 — but only as well as the (cold-started) bias already works, so it's a
   symptom of (1)/(2), not an independent cause.

Bidirectional prefix attention should be a pure *advantage* for the graph
arm (full context mixing vs. flat's causal buildup) and it isn't showing up
in the numbers — further evidence (1)/(2) dominate over anything structural
being "missing."

## At a glance

Every row reuses an existing 3-seed result as its control instead of
re-running it — see each section for the exact source run/numbers.

| # | Experiment | Config | New runs | Control (reused) | New-arm result | Verdict |
|---|---|---|---|---:|---:|---|
| E1 | Flat context-order shuffle | `030_flat_shuffle_diag.jsonc` | 3 | 0.7490 ± 0.0003 | **0.7525 ± 0.0035** | premise falsified — no cost to shuffling |
| E2 | `magnetic_dim` capacity | `031_magnetic_dim_sweep.jsonc` | 9 | 0.7351 ± 0.0076 (dim 128) | 0.7382 / 0.7347 / 0.7300 (dims 32/64/256) | capacity ruled out — flat, no trend |
| E3 | SPD-depth position encoding | `032_node_position_spd_depth.jsonc` | 3 | 0.7351 ± 0.0076 | **0.6412 ± 0.0037** | fix rejected — clear regression (−9.4 F1) |

**Total: 15 new training runs**, all WebQSP, all 3 seeds, all complete. See
each experiment's own Result subsection below for the full per-seed tables,
Hits@1/Hit breakdowns, and reasoning.

---

## E1 — Flat context-order shuffle diagnostic

**Question:** does flat's advantage come (partly) from the retrieval-order
→ RoPE-position correlation, or is it purely a content/representation
effect? This doesn't fix anything — it's a cheap sanity check that derisks
E3 before we invest in the collator rewrite. If shuffling tanks flat's F1,
positional/order signal is load-bearing and E3's premise is strong. If it
barely moves, the flat advantage is elsewhere and E3 alone won't close the
gap.

**Design.** Add `flat_shuffle_lines: bool = False` to `RunConfig`
(data-prep key, since it changes what's baked into the flat `.jsonl`
cache). In `flat_data.py::build_flat_rows`, right after `lines =
lines_fn(record, entity_names, cfg)` (shared across all `versions` of one
question, same as today — only the *answer* order varies per version, not
the triple order), shuffle `lines` in place with the same `rng` already
threaded through the function when the flag is set. One extra
`rng.shuffle(lines)` call, gated by the flag — reuses the existing
`random.Random(f"{data_seed}:{split}")` stream so the run stays
reproducible. `flat_data_config_key` gets a `_shuf` suffix when true (mirrors
the `_cvt` / `_nocvt` suffix convention), so this is a genuinely new cache,
not a mutation of the existing one — the unshuffled cache and its 3 seeds of
results stay valid as the control.

`validate()`: reject `flat_shuffle_lines=True` outside `flat_data_prep` /
`flat_train` modes (mirrors the existing `question_node` graph-only guard in
reverse).

**Run.** Mirror `021_webqsp_recipe_refresh.jsonc`'s flat arm exactly (`mode:
flat_train`, `cvt_collapse: true`, `lora_dropout: 0.15` — the recipe that
also anchors E2/E3's graph control, so all three experiments read off one
consistent "frozen recipe" lineage rather than mixing pre-/post-regularization-
campaign numbers) with `flat_shuffle_lines=True`, 3 seeds. **Do not re-run
the control** — the 3 existing seeds are already on record
(`results/webqsp_recipe_refresh/runs.jsonl`: seed0 0.74937, seed1 0.74875,
seed2 0.74895, mean 0.7490 ± 0.0003); only the 3 shuffled-order runs are new.
(Earlier drafts of this doc cited `collapse_2x2`'s 74.93±0.23 as the flat
reference — that sweep predates the `lora_dropout=0.15` regularization fix
and is superseded by 021's 0.7490±0.0003 for anything meant to compare
against the current frozen recipe.)

**Reading it.** Compare shuffled vs. unshuffled mean F1 against the ±0.2–0.3
seed-noise bar established across prior sweeps (e.g. `bias_lr_webqsp`,
`capacity_lora`). A drop well past that bar validates E3's premise; a wash
means the position/order signal isn't where flat's edge comes from and E3
should be deprioritized pending a rethink.

- [x] Add `flat_shuffle_lines` field + CLI flag + `validate()` guard
- [x] Implement the shuffle in `build_flat_rows` (+ cache-key suffix). **Bug
      caught by a sanity diff before launch**: the first implementation drew
      the line-shuffle from the same `rng` stream the answer-order
      augmentation consumes, which shifted every subsequent `rng.shuffle(order)`
      draw — `target` differed between shuffled/unshuffled at identical seeds,
      confounding the diagnostic. Fixed with an independent `line_rng` stream
      (`f"{data_seed}:{split}:lines"`); regression test added
      (`test_target_order_is_unaffected_by_line_shuffle`).
- [x] 8 new tests in `tests/test_kgqa_flat_shuffle.py`, full suite green
      (237 passed / 11 skipped, no regressions)
- [x] Data prep built (`--mode flat_data_prep --flat-shuffle-lines`, CPU,
      ~6s): 2607/246/1628 train/dev/test questions — identical counts to the
      unshuffled cache.
- [x] Verified against the existing on-disk unshuffled cache: same question
      order, same `gold_answers`, and (after the rng-independence fix) same
      `target` per row; triple-line **content** matches to 99.99% (9/95,969
      lines across 9/1628 test questions differ) — traced to the on-disk
      unshuffled cache predating a minor `entities_names.json` update
      (unrelated to this experiment: any fresh build today vs. that
      already-frozen cache would show the same handful of newly-resolved
      names). Accepted as noise — 0.009% of lines, far under the seed-noise
      floor this whole plan is built around; not worth rebuilding+retraining
      the reused control over.
- [x] `030_flat_shuffle_diag.jsonc`: **3 new runs only**
      (`flat_shuffle_lines=True` × seeds{0,1,2}), everything else pinned to
      the 021 flat recipe — no control arm in this sweep, `021` already has it
- [x] Launched as job **112750** (2026-07-16); read result against the 3
      existing `webqsp_recipe_refresh` flat seeds once it completes

### Result (complete, 2026-07-16)

| arm | seed 0 | seed 1 | seed 2 | mean | Hits@1 (mean) | Hit (mean) |
|---|---:|---:|---:|---:|---:|---:|
| unshuffled (021, control, reused) | 0.74937 | 0.74875 | 0.74895 | **0.7490 ± 0.0003** | — | — |
| **shuffled (`flat_shuffle_lines=True`)** | 0.7495 | 0.7509 | 0.7573 | **0.7525 ± 0.0035** | 0.8036 | 0.8464 |

**Verdict: the premise is falsified — scrambling flat's triple order costs
nothing.** Shuffled mean is +0.35 pts *above* the control, well within (in
fact on the favorable side of) combined noise; the shuffled arm's own spread
(±0.0035) is wider than the tight unshuffled control (±0.0003) but the
central estimate doesn't move against the hypothesis in either practical
direction. Flat's edge over the graph arm is **not** explained by the
retrieval-order → RoPE-position correlation — whatever flat is doing better,
it isn't leaning on serial position as a distance proxy. This retroactively
weakens the premise E3 was built on (see E3's result below, which independently
confirms the position-encoding fix doesn't help — consistent with there being
no real order signal for a RoPE-position mechanism to recover in the first
place).

---

## E2 — `magnetic_dim` capacity sweep

**Question:** is the bias channel's *width* (not its position-encoding
deficit) a binding constraint? Lower prior than E1/E3 — the graph-bias
weights already converge well and show message-passing-like attention
patterns in other benchmarks — but it's a pure hyperparameter sweep on
existing code, so it's nearly free to rule in/out alongside the other two.

**Design.** No code change. `magnetic_dim` (the magnetic-bias MLP hidden
width, currently 128) is a model-architecture key, **not** part of
`data_config_key` — sweeping it reuses the existing `.gtds` cache built for
the question-node `isolated` arm.

**Run.** `magnetic_dim ∈ {32, 64, 256}` × seeds `{0,1,2}` (9 new runs) on the
frozen best graph recipe (`029_question_node_webqsp.jsonc`'s `isolated` mode:
r64, bias_lr 5e-3, k_hop 0, question_node=isolated). **Do not re-run
`magnetic_dim=128`** — it already has 3 seeds on record (0.7351 ± 0.0076 test
F1, `results/question_node_webqsp/report.md`); reuse that as the control
point on the grid.

**Reading it.** 3 seeds/cell across the whole grid, consistent with every
other arm in this plan. If a direction shows a real effect (monotonic with
width, or a knee), that's a clean, trustworthy signal already at 3 seeds — no
follow-up round needed either way.

- [x] `031_magnetic_dim_sweep.jsonc`: `magnetic_dim ∈ {32,64,256}` × seeds
      `{0,1,2}` (9 runs), base = `029_question_node_webqsp.jsonc` `isolated`
      arm — no `magnetic_dim=128` in this sweep, it's already on record
- [x] Launched as job **112751** (2026-07-16), no data prep needed
- [x] Read against the existing 128-dim / 3-seed control — see Result below.

### Result (complete, 2026-07-16)

| magnetic_dim | seed 0 | seed 1 | seed 2 | mean F1 | std | Hits@1 (mean) | Hit (mean) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.7356 | 0.7429 | 0.7359 | 0.7382 | 0.0034 | 0.7838 | 0.8350 |
| 64 | 0.7360 | 0.7350 | 0.7332 | 0.7347 | 0.0012 | 0.7821 | 0.8288 |
| **128 (control, reused)** | 0.7381 | 0.7265 | 0.7407 | **0.7351** | 0.0076 | 0.7803 | 0.8325 |
| 256 | 0.7221 | 0.7336 | 0.7344 | 0.7300 | 0.0056 | 0.7797 | 0.8305 |

**Verdict: capacity ruled out, as expected.** All four widths land in a tight
0.730–0.738 band — every pairwise gap is smaller than or comparable to the
control's own seed noise (±0.0076), and there's no clean monotonic trend (32
is nominally highest, 256 nominally lowest, but 64 sits *below* 32 despite
being wider, which is exactly the shape of noise, not a real width effect).
Confirms the prior stated going in: the bias channel isn't capacity-starved
in the width dimension. Combined with the graph-bias weights already
converging well and resembling message-passing patterns in other benchmarks,
this closes the "make the bias MLP bigger" branch — not worth revisiting
without new evidence pointing at capacity specifically.

---

## E3 — SPD-depth structural position encoding (the main bet)

**Question:** does giving prefix nodes a RoPE-visible, graph-structure-derived
position (instead of resetting every node to 0) recover the distance signal
flat gets implicitly from retrieval order? This is the one that could
actually flip the flat/graph gap, not just narrow it.

### Design

Replace the per-node `arange(len)` reset in
`text_graph_collator_v2.py::_pack_one` with a depth-banded scheme:

```
position_ids(token i of node v) = STRIDE * depth(v) + i
```

where `depth(v) = shortest_path_dists[prompt_node, v]` (already computed and
cached per graph for the SPD bias — just a row-read, no new feature), capped
at some `DEPTH_CAP` so a pathological longest-shortest-path doesn't push
positions outside RoPE's well-trained range (Llama-3.2's RoPE handles this
comfortably at the scale we need — `max_nodes=512` nodes × a handful of
tokens each — but pin a cap rather than leaving it unbounded). Unreachable
nodes (SPD sentinel) share the cap's band.

**The prompt node is defined as one band past the deepest prefix node**
(`depth(prompt) := DEPTH_CAP_OBSERVED + 1`, not 0) — this generalizes today's
"prompt packed last in the token sequence" convention into position space
too, so the prompt's tokens (the generation query) sit at strictly larger
RoPE positions than any prefix node, preserving the intuitive "prompt reads
last" structure. With **only** a prompt node (no prefix), this depth is
vacuous by construction and the scheme collapses to `position_ids =
arange(len)` — bit-identical to today.

`prepare_inputs_for_generation`'s position-id extension (`causal_lm.py:317-339`,
"newly generated tokens continue the prompt node's local counter") needs **no
change** — it only reads `position_ids[:, -1]` and increments, which works
under any position scheme, not just the current reset-to-0 one.

Gate behind a new `RunConfig.node_position_mode: str = "reset"` field
(`NODE_POSITION_MODES = ("reset", "spd_depth")`, mirrors the
`QUESTION_NODE_MODES` pattern) so `"reset"` stays the default and every
existing config/checkpoint is untouched. `"spd_depth"` requires `cfg.spd=True`
(needs `shortest_path_dists` on the batch) — add a `validate()` check.
**Not** part of `data_config_key`: this only changes how the collator packs
positions at train time, not the cached graph features, so it reuses the
existing `.gtds` cache for whatever recipe it's crossed onto — no data prep
needed before training.

Open implementation decisions (pin during coding, not before): exact
`STRIDE` and `DEPTH_CAP` constants. `STRIDE` just needs to exceed the
per-node token count (mean ~3, p99 well under 20 per the README's built-split
token-length tables), so something like 32–64 leaves comfortable headroom;
`DEPTH_CAP` can likely reuse `max_spd` truncation already used by `SPDBias`
rather than inventing a second cutoff.

### Required tests before any training run

Two invariants must hold for the new mode, verified with actual tests
against the **current v2 stack** — the only existing check
(`src/experiments/permutation_equivariance/__main__.py`) is a standalone
script against the legacy v0 model, not a pytest, and doesn't cover position
encoding specifically:

- [x] **Backward compatibility**: a single-node (prompt-only) `TextGraph`
      through `GraphCollatorV2(node_position_mode="spd_depth")` produces
      `position_ids == arange(len)`, identical to `node_position_mode="reset"`
      and to plain tokenization.
- [x] **Permutation equivariance**: build the same logical graph twice with
      two different input node orderings (a 5-node star fixture in the style
      of the legacy experiment's `ADJ_MATRIX`), run both through the collator +
      model forward under `"spd_depth"`, assert identical logits up to node
      correspondence. Passed to float64 precision (~1e-8 max diff, numerical
      noise floor) — see `tests/test_node_position_encoding.py`.
- [x] Existing collator/structural-mask/flex test suite stays green
      (`test_graph_bias.py`, `test_collator_bucketing.py`, `test_flex_cpu.py`) —
      full suite: 247 passed / 11 skipped, 0 regressions.

### Run

Only `node_position_mode=spd_depth × seeds{0,1,2}` gets trained (3 new runs).
**`reset` is NOT re-run** — `029_question_node_webqsp.jsonc`'s `isolated` arm
(0.7351 ± 0.0076 test F1) is the control, valid as long as `"reset"` mode is
proven to be a byte-identical no-op against the pre-change collator code path
— which is a **code-level equivalence test**, not a retrain (see the test
list above: comparing `GraphCollatorV2` output tensors old-code-path vs.
`node_position_mode="reset"` on the same batch is exact and free, so there is
no need to burn a GPU run "just to check" — a passing test *is* the
verification here).

**If it wins:** stack it onto the current best graph recipe and rerun the
full dev/test comparison against flat's 74.93–75.03 F1 headline to see
whether the gap closes or flips. Also worth re-crossing with E2's
`magnetic_dim` finding (if E2 showed anything) and re-checking whether the
024 bias-ablation ranking (`magnetic` > `spd` > `none`) still holds once
position carries some of the load SPD alone was carrying before.

- [x] Add `node_position_mode` field + `NODE_POSITION_MODES` + CLI flag +
      `validate()` guard (requires `spd=True`, graph-arm only)
- [x] Implement depth-banded position assignment: `_node_base_offsets` in
      `text_graph_collator_v2.py` — `depth(v) = min(spd[prompt, v], max_spd)`
      (reuses the existing `max_spd` knob, same far/unreachable bucket
      `SPDBias` clamps into — no new constant introduced), `depth(prompt) =
      max(prefix depths) + 1`, `STRIDE` computed per-graph as the longest
      node's token count (always safe, no cross-module constant to keep in
      sync). `prepare_inputs_for_generation` needed **no change** — confirmed
      by inspection, it only reads/extends `position_ids[:, -1]`, agnostic to
      how the base positions were assigned.
- [x] Backward-compatibility test (`test_single_node_graph_is_plain_arange_under_both_modes`,
      `tests/test_node_position_encoding.py`) — passes for both modes.
- [x] Permutation-equivariance test
      (`test_permutation_equivariance_of_spd_depth_positions`) — built a
      5-node star graph (4 prefix + 1 prompt, synthetic tiny `GTLMLlamaConfig`,
      spd-only bias, real BFS-computed SPD via networkx) in two different
      prefix-node labelings, ran both through the actual v2 model forward in
      float64, compared logits at the prompt's token positions: **max abs
      diff 3.7e-8** (float64 summation-order noise; the invariant holds).
      A same-fixture negative check confirms the two builds genuinely differ
      going in (not a vacuous test).
- [x] `"reset"`-mode no-op equivalence test (`test_reset_mode_is_a_noop`) —
      byte-identical to the pre-change hand-computed formula and to the
      constructor-default (mode omitted). Licenses reusing `029`'s
      isolated-arm numbers (0.7351 ± 0.0076) as this experiment's control,
      no retrain needed.
- [x] Full test suite green: 247 passed / 11 skipped, 0 failures/regressions.
- [x] Real-data smoke check (no GPU available on the dev node, so a full
      `Trainer.train()` CPU run wasn't practical — it timed out at 5 min
      without finishing 2 steps on a 1B model). Instead: loaded 8 real
      examples (N=16 to N=502 nodes) straight from the existing `qnisolated`
      `.gtds` cache and ran them through `GraphCollatorV2(node_position_mode=
      "spd_depth")` directly — no crashes, positions numerically sane (e.g.
      N=502 example: max position 11351, consistent with a ~175-token widest
      node × depth up to `max_spd+1`=65 — well inside Llama-3.2's RoPE range).
      This exercises exactly the part the synthetic unit tests couldn't (real
      SPD tensors/sentinels at real scale) without needing a GPU.
- [x] `032_node_position_spd_depth.jsonc`: **3 new runs only**
      (`node_position_mode=spd_depth` × seeds{0,1,2}) — no `reset` arm in
      this sweep, `029` already has it. No data prep needed (reuses the
      `qnisolated` `.gtds` cache — `node_position_mode` isn't in the cache
      key). Launched as job **112768** (2026-07-16).
- [x] Read against flat's headline (021: 0.7490 ± 0.0003) and the `reset`
      control (029 isolated: 0.7351 ± 0.0076) — see Result below.

### Result (complete, 2026-07-16)

| arm | seed 0 | seed 1 | seed 2 | mean F1 | Hits@1 (mean) | Hit (mean) |
|---|---:|---:|---:|---:|---:|---:|
| `reset` (029 isolated, control, reused) | 0.7381 | 0.7265 | 0.7407 | **0.7351 ± 0.0076** | 0.7803 | 0.8325 |
| **`spd_depth`** | 0.6402 | 0.6373 | 0.6461 | **0.6412 ± 0.0037** | 0.7827 | 0.8016 |

(Control values re-verified directly from `results/question_node_webqsp/runs.jsonl` — 0.73807/0.72648/0.74074, mean 0.7351, matching the README's reported ±0.0076; an earlier draft of this table briefly had the wrong control numbers pasted in from a different sweep and was corrected before this was finalized.)

**Verdict: `spd_depth` is a clear, consistent regression — reject this fix.**
−9.4 F1 points, and *more* consistent across seeds (±0.0037) than the control
itself (±0.0076) — this is a robust effect, not a bad seed. The revealing
detail is **where** the damage lands: **Hits@1 is essentially unchanged**
(0.7827 vs. 0.7803 — if anything marginally higher) while **F1 collapses**.
Top-1 answer selection is intact; multi-answer *recall* is what breaks. That
localizes the failure mode more precisely than "RoPE distance decay weakens
attention into the graph generally" (which should have hurt Hits@1 too) —
it points instead at something that specifically degrades the model's ability
to enumerate the *full* answer set once graph nodes are spread across large,
STRIDE-scaled position gaps: plausibly the sheer position values (into the
thousands–tens of thousands for large subgraphs, vs. single digits under
`reset`) push some fraction of prefix nodes into a RoPE regime the model
handles less reliably for multi-hop aggregation specifically, even if it can
still latch onto *one* good answer. Combined with E1's null result (flat's
edge isn't about order/position at all), the honest reading is that
**RoPE-visible relative position was never the missing ingredient** — the
gap has to be explained some other way. `node_position_mode` stays
implemented (real, tested, reversible — see the invariant tests) but its
default (`"reset"`) is the one to keep; `"spd_depth"` is a documented negative
result, not a recommended arm.

**Post-mortem diagnosis (2026-07-17, measured on 120 real `qnisolated`
train items).** The regression has a concrete, largely self-inflicted
mechanism — an interaction between `spd_depth` and `question_node=isolated`
that the invariant tests couldn't catch:

1. **The isolated QUESTION node pins the prompt at band 65 on every single
   item.** Unreachable SPD is stored as 32767 (int16 ∞) and
   `min(spd, max_spd)` clamps it to 64, so the disconnected QUESTION node
   always lands at depth 64 → `depth(prompt) = max+1 = 65`, in **120/120
   sampled items** — while the *actual* content depths are 1–5 (mode 3).
   Bands 6–63 are empty on every example.
2. **The intended signal is therefore ~zero and the side effect is huge.**
   The discriminative part (depth-1 vs. depth-5 nodes differ by 4·STRIDE)
   rides on a near-constant ~64·STRIDE offset between the prompt/generation
   region and *all* content — a ≤6% relative difference. Measured RoPE gap
   from prompt to the shallowest (answer-bearing) nodes: min/median/max =
   **384 / 768 / 16384** positions, vs. ~tens under `reset`. The model got
   almost no usable structure signal, but every generated token now attends
   to entity names ~10²–10⁴ positions away instead of ~10¹.
3. **Per-item STRIDE varies 6→256 tokens (40×)** (max node length per
   graph), so "one band" means a different physical distance on every
   example — whatever residual signal existed was also scale-inconsistent.
4. **RoPE can't be trained away.** Unlike E2's additive bias tables (which
   can learn ≈0 and produce a clean null), rotary displacement is baked
   into every head — a harmful position signal *hurts* rather than washes
   out. That's why E3 regressed while E2 nulled.

This fits the Hits@1-flat/F1-collapse signature: top-1 extraction survives
(the structural-bias channel still delivers relevance; one strong answer is
findable even through weak long-range attention), but enumerating the full
answer set requires repeated reliable copying of entity-name tokens from
positions hundreds–thousands away, which is exactly what degrades. Caveat:
this makes E3 a rejection of *this implementation*, not a fully clean
falsification of the RoPE-position hypothesis — a corrected variant (depth
of unreachable nodes = 0 or excluded from the prompt-band max; prompt at
band 0 so *shallow = near*; small global stride) was never tested. Given
E1 independently showed order/position isn't flat's edge, the corrected
variant is low-priority, but the two results should not be conflated.

---

## Execution order (as it actually happened)

E1 and E2 needed no new code beyond a one-line flag / an existing knob and
launched first (~15:07), both as sbatch array jobs. E3's implementation
(collator change + the two invariant tests) happened in parallel while those
ran; it launched ~6 min later (~15:13) once the tests were green, and then
sat queued for ~40 min behind unrelated priority scheduling before starting.
E1 finished first (flat is the fast arm, ~2h), landing its 3 seeds well
before E2 or E3 — which is what let E1's null result inform how E3's result
(finishing a bit later) got read: E1 already showed the premise behind E3
was weak before E3's own negative result confirmed it independently. E2 was
the slowest to fully close out (9 runs sharing GPUs with E1, ~4.5h for the
last cell) but never blocked reading E1/E3. All three closed out same-day,
~19:45.
