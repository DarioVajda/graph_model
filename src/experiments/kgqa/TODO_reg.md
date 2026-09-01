# TODO_reg — Regularization knobs + probe sweep for the GTLM overfitting question

Status: **COMPLETE** (started 2026-07-11; all 32 training runs + all
train-slice evals finished 2026-07-12). Verdict + keep/remove decision below
(§ Final verdict). Experiment-as-run committed as 06ac9bd, tag
`reg-probes-2026-07`; mechanism removal is the follow-up commit.

## Summary — every regularization arm (all rounds)

Control = frozen nmax50 recipe, 3 seeds: test F1 **72.55 ± 0.22**, train-slice
F1 **96.71**. Test = GNN-RAG F1, full split; train-slice = same 200 sampled
train questions everywhere (internally consistent rescoring). Sweeps:
R1 = 015 reg_probes, C = 016 reg_combo, R2 = 017 reg_round2.

| round | arm | knobs (nonzero only) | test F1 by seed | mean (Δ ctrl) | train-slice | verdict |
|---|---|---|---|---|---|---|
| — | control | lora_dropout 0.05 (recipe default) | 72.35 / 72.86 / 72.45 | 72.55 | 96.71 | baseline |
| R1 | lora_do | lora_dropout 0.15 | 72.95 / 73.39 | **73.17 (+0.62)** | 96.83 | **only winner**; train-fit unchanged → better adaptation, not de-memorization |
| R2 | lora25 | lora_dropout 0.25 | 71.95 / 71.95 / 71.30 | 71.73 (−0.82) | 95.96 | overshoots — starts cutting train-fit AND test; dose peaks near 0.15 |
| R1 | wd | bias_weight_decay 0.1 | 72.57 / 73.02 / 72.02 | 72.54 (−0.01) | 96.53 | flat — no effect anywhere |
| R2 | wd3 | bias_weight_decay 0.3 | 71.51 / 71.97 / 73.08 | 72.19 (−0.36) | 96.65 | train-fit untouched, test drifts down |
| R1 | eig | eigvec 0.1 + mlp 0.1, per-layer masks | 58.50 / 66.27 / 57.31 | 60.69 (−11.86) | 89.09 | overdosed + incoherent spectrum |
| R2 | eig_shared | eigvec 0.05, ONE mask/forward | 72.27 / 71.69 / 72.90 | 72.28 (−0.27) | 96.41 | "recovery" = the dose stopped doing ANYTHING (train-fit ≈ control) — no benign regularization window found |
| R2 | eig_shared_onset | + reg_onset_frac 0.33 | 72.30 / 72.00 / 71.63 | 71.98 (−0.57) | 97.08 | late onset memorizes slightly MORE than control |
| R1 | droppath | bias_droppath 0.1 | 68.21 / 69.83 / 67.83 | 68.62 (−3.93) | 94.16 | drops generalizing signal |
| R2 | droppath05 | bias_droppath 0.05 | 69.69 / 71.48 / 71.03 | 70.73 (−1.82) | 96.06 | half dose ≈ half damage |
| R2 | elem | bias_dropout 0.1 (element-wise) | 33.09 / 40.82 / 38.16 | 37.36 (−35.19) | 64.10 | catastrophic: logit-space zero+rescale corrupts attention |
| C | combo | all R1 knobs at once | 52.03 / 64.21 / 51.97 | 56.07 (−16.48) | 84.51 | compounded eig damage; relocation unreadable |

Reading order: rows grouped by mechanism family (LoRA dropout, bias weight
decay, magnetic spectral dropout, whole-bias droppath, element-wise bias
dropout, everything combined) — each family's round-2 row is the dose/mechanism
correction of its round-1 row. Train-slice jobs: controls 110940, R1 111023,
R2 111104, combo 111105 (same 200 train questions throughout).

**The campaign-wide pattern (the strongest single result):** sorting all 12
rows by train-slice vs test, there is NO arm that cut train-fit while holding
test — every mechanism either left train-fit at ~96–97 (wd, wd3, eig_shared,
lora_do) or dragged test down faster than train-fit (droppath 2.5↓/3.9↓,
eig 7.6↓/11.9↓, elem 32.6↓/35.2↓, combo 12.2↓/16.5↓, lora25 0.75↓/0.82↓).
The "de-memorize the graph channel" hypothesis is dead across 8 mechanisms ×
2 doses: memorization headroom on this dataset is not convertible into test F1
by ANY tested regularizer — only more data (CWQ arm) can move it.

## Progress

- [x] Part 1 — weight-decay fix (`bias_weight_decay` knob, shape-based split,
  `SPDBias.weights` tagged `_no_weight_decay`; default 0.0 = historical math).
- [x] Part 2 — bias-path dropouts (`magnetic_eigvec_dropout`, `magnetic_mlp_dropout`,
  `bias_droppath`; all functional, droppath before the k-hop gate). Eigvec rescale
  implemented as `1/sqrt(1-p)` on kept V columns (product carries `1/(1-p)` —
  same math as the plan's (c), one shared code path for folded/legacy).
- [x] Part 3 — `lora_dropout` knob (`select_active_params` already honored the
  dict key; both arms consume `cfg.lora_config()`).
- [x] Part 4 — KGQA plumbing (RunConfig fields, `bias_params()` nonzero-only,
  5 CLI flags, `validate()` range checks, trainer threading, run-record fields).
  Note: KGQA logs via `_save_train_record`, not `save_run_metadata` — knobs added
  there (plan intent, actual code).
- [x] Part 5 — tests: 12 new (decay grouping + no-regression, eigvec dropout
  eval-noop/truncation-equivalence/MC-rescale, MLP dropout + folded-legacy parity
  under dropout, droppath all-or-nothing + gate survival, ckpt+dropout grad
  parity, flag round-trips + validate rejections). Full suite: 209 passed, 11
  skipped (GPU-only), 0 failures.
- [x] Part 6 — probe sweep `015_regularization_probes.jsonc` (sweep name
  `reg_probes`, 11 runs, B300, max_concurrent 11 to stay under the ≤16 cap).
  Expansion + flag round-trip dry-checked (cache key matches the existing
  nmax50 cache). GPU smoke passed first (all knobs at once, 20 steps, flex +
  grad-ckpt, job 110932, COMPLETED 0:0; artifacts removed). Sweep launched as
  array job **110933** (2026-07-11 ~16:05). Control train-slice job **110940**
  (3 frozen-recipe checkpoints, `train_slice_probes.py`) launched in parallel.
- [x] Part 6b — combo arm added (user question 2026-07-11: uncrossed arms can't
  detect graph<->LoRA memorization relocation, where no single knob moves test
  F1 but squeezing both channels at once does). `016_reg_combo.jsonc` = ALL
  knobs on (wd 0.1 + eig/mlp 0.1 + droppath 0.1 + lora_do 0.15), 3 seeds,
  array job **110947**. Total concurrent: 11 + 3 = 14 ≤ 16.
- [x] Part 7 — readout: per-arm metrics + train-slice eval + verdict.
  Train-slice jobs: controls 110940, round-1 111023, round-2 111104 (all 18
  checkpoints), combo 111105. Full grid in the summary table at the top.
  Tooling ready: `error_analysis/train_slice_probes.py` (parameterized version
  of train_slice_eval.py; same 200 questions; controls rescored with the same
  script on the nmax50 control checkpoints for internal consistency).
  Controls (job 110940): train-slice F1 0.9656/0.9689/0.9669, Hits@1 ≈ 1.0 —
  reproduces the 96.7 memorization baseline. Probe-arm train-slice pending the
  last round-1 checkpoints.

## Round-1 results (015 reg_probes + 016 reg_combo, jobs 110933/110947) — FINAL

Control: test F1 **72.55 ± 0.22** (retune_v3_0006 s0 72.35, frozen_nmax50 s1
72.86, s2 72.45). All numbers = GNN-RAG test F1, full split.

| arm | knobs | test F1 by seed | mean Δ | verdict |
|---|---|---|---|---|
| lora_do | lora_dropout 0.15 | 72.95, 73.39 | **+0.62** | ONLY arm above control — both seeds beat every control seed. Generic LoRA overfit confirmed as a real lever. |
| wd | bias_weight_decay 0.1 | 72.57, 73.02, 72.02 | ≈ 0 (72.54) | harmless but flat at 0.1 |
| droppath | bias_droppath 0.1 | 68.21, 69.83, 67.83 | −3.9 | overdosed: losing the whole channel ~1.6 layers/step degrades real signal |
| eig | eigvec 0.1 + mlp 0.1 | 58.50, 66.27, 57.31 | −11.9 | heavily overdosed + per-layer independent masks gave the net no consistent spectrum; also unattributable (bundled with MLP dropout) |
| combo | all knobs | 52.03, 64.21, 51.97 | −16.5 (56.07) | inherited + compounded the eig damage as expected; no graph↔LoRA relocation signal is readable through it |

Interpretation so far: the graph-channel dropouts at 0.1 corrupt a load-bearing
signal (the answer IS a subgraph node — the magnetic bias does real work
locating it), rather than just suppressing memorization. The productive
direction found by round 1 is the LoRA channel.

**Round-1 train-slice (job 111023, same 200 questions as controls):**

| arm | train-slice F1 (seeds) | mean | test mean | grid verdict |
|---|---|---|---|---|
| control | 96.56 / 96.89 / 96.69 | 96.7 | 72.55 | memorization baseline |
| wd 0.1 | 96.25 / 96.81 / 96.52 | 96.5 | 72.54 | no effect anywhere — 0.1 too weak |
| lora_do 0.15 | 96.88 / 96.78 | 96.8 | 73.17 | test ↑ with train-fit UNCHANGED: not de-memorization — better-quality adaptation (noise-robust adapters) |
| droppath 0.1 | 94.98 / 94.31 / 93.20 | 94.2 | 68.62 | train-fit ↓2.5 but test ↓3.9: removed generalizing signal |
| eig 0.1 | 85.52 / 91.53 / 90.22 | 89.1 | 60.69 | train-fit ↓7.6 but test ↓11.9: same, worse |

Key Part-7 conclusion from round 1: the graph-bias dropouts DID cut
memorization exactly as hypothesized (train-fit → ~89–94), but test fell
harder — the structural channel's content is disproportionately *generalizing*
signal, so memorization there is not the binding constraint. The LoRA channel
is the lever that moves test F1.

## Round-2 mechanisms (implemented 2026-07-11, all tested; suite 219 passed)

- [x] `bias_dropout` — element-wise dropout on the summed (H,N,N) bias
  (user-suggested granularity; gentlest corruption, channel always present).
- [x] `magnetic_eigvec_shared_mask` — ONE eigvec keep-mask per forward, sampled
  in the causal-LM mixin (outside checkpointed layers) and threaded via
  `GraphContext.features["magnetic_keep"]`; all layers see the same spectral
  truncation (round-1 eig sampled per layer per step = incoherent corruption).
- [x] `reg_onset_frac` — `RegOnsetCallback` keeps bias dropouts at 0 for the
  first fraction of training (zero-init channel forms cleanly), then switches
  the configured rates on. Attribute-flip only; stateless across resumes.

## Round-2 sweep (`017_reg_round2.jsonc`, 18 runs, B300, max_concurrent 16)

| arm | knobs | hypothesis |
|---|---|---|
| lora25 | lora_dropout 0.25 | dose-response of the working lever |
| wd3 | bias_weight_decay 0.3 | wd 0.1 flat → one stronger probe |
| eig_shared | eigvec 0.05 shared-mask, no MLP dropout | coherent + halved + unbundled fixes the eig arm |
| eig_shared_onset | + reg_onset_frac 0.33 | regularize late, after channel formation |
| droppath05 | bias_droppath 0.05 | halved dose |
| elem | bias_dropout 0.1 | element-wise variant |

- [x] Round-2 sweep launched: smoke 111007 COMPLETED 0:0 (shared mask under
  flex+ckpt, onset flip mid-run, elem dropout), then array job **111008**
  (18 runs, 2026-07-11 ~22:55).
- [x] Round-1 train-slice on probe checkpoints (job 111023 — tables above).
- [x] Round-2 sweep complete (all 18 runs COMPLETED 2026-07-12, results below).
- [x] Round-2 train-slice on all 18 checkpoints: job **111104**; combo
  checkpoints (missed by 111023) backfilled via job **111105**.
- [x] Final readout: full grid + campaign-wide pattern in the summary table.

## Round-2 results (017 reg_round2, job 111008) — test F1 FINAL

Same control (72.55 ± 0.22). **No round-2 arm beats control.**

| arm | knobs | test F1 by seed (s0/s1/s2) | mean Δ | verdict |
|---|---|---|---|---|
| lora25 | lora_dropout 0.25 | 71.95 / 71.95 / 71.30 | −0.82 (71.73) | 0.25 OVERSHOOTS — 1.4 below the 0.15 arm. Dose-response peaks near 0.15. |
| wd3 | bias_weight_decay 0.3 | 71.51 / 71.97 / 73.08 | −0.36 (72.19) | stronger decay drifts negative; 0.1 flat + 0.3 mildly harmful → no decay setting helps. Fix stays, value stays 0. |
| eig_shared | eigvec 0.05, shared mask | 72.27 / 71.69 / 72.90 | −0.27 (72.28) | coherent + halved + unbundled recovers the round-1 −11.9 almost fully — but only back TO control, never above |
| eig_shared_onset | + onset 0.33 | 72.30 / 72.00 / 71.63 | −0.57 (71.98) | late onset does not help (slightly worse than immediate) |
| droppath05 | bias_droppath 0.05 | 69.69 / 71.48 / 71.03 | −1.82 (70.73) | half dose ≈ half damage — channel-dropping hurts at every dose |
| elem | bias_dropout 0.1 | 33.09 / 40.82 / 38.16 | **−35.2** (37.36) | CATASTROPHIC — worst arm of the campaign. Runs trained all 15 epochs, no divergence (loss 0.2–0.7, no NaN): the mechanism itself is destructive. Element-wise zero+rescale on attention *logits* corrupts individual attention distributions every step (unlike activation dropout, softmax is not linear in the logits, so the 1/(1−p) rescale does not preserve expected attention), and eval sees an unscaled bias the model never trained with. |

## Final verdict (2026-07-12)

32 training runs (11 + 3 + 18) over 8 regularization mechanisms, 2 doses each
for the main ones, vs a 3-seed control. Outcome of the campaign:

1. **The graph channel is exonerated as the binding constraint.** Every
   graph-side regularizer at every dose — per-layer or coherent (shared mask),
   immediate or late-onset, whole-channel (droppath) or element-wise — lands at
   best AT control (eig_shared −0.27) and at worst −35. The round-1 train-slice
   shows why: these knobs DO cut memorization (train-fit 96.7 → 89–94) but test
   falls *harder* — the structural channel's content is disproportionately
   generalizing signal (the answer is a subgraph node; the magnetic bias does
   real work locating it). The round-2 train-slice closes the loop: the
   "fixed" gentle dose (eig_shared 0.05) recovered test only by ceasing to
   affect train-fit at all (96.4 ≈ control) — there is no dose window where a
   graph-side regularizer trades memorization for test F1.
2. **LoRA dropout 0.15 is the single validated positive** (+0.62, both seeds
   above every control seed) and the dose-response peaks there (0.25 → −0.82).
   Train-fit unchanged at 96.8 → it is better-quality adaptation, not
   de-memorization.
3. **Memorization lives in the LoRA/backbone channel** (Part-7 grid, branch 2);
   the CWQ data arm — not regularization — is the lever for the 18-pt
   generalization gap.

**Actionables carried forward** (into D5 scale recipe + CWQ arm):
- `lora_dropout: 0.15` (re-confirm dose on CWQ if tuning budget allows: the
  +0.62 was measured at 15 epochs on ~3k questions, where LoRA overfit pressure
  is maximal; with ~10× data it may shrink).
- Weight-decay fix (shape-based split + `_no_weight_decay` tag) stays as
  hygiene; `bias_weight_decay` stays **0.0** (0.1 flat, 0.3 harmful).
- All graph-bias regularizers stay 0 on CWQ; do not re-sweep them.
- Rethink `num_epochs` for CWQ from token budget, not inherited 15 (15 epochs
  on WebQSP ≈ the memorization regime CWQ is supposed to escape).
- Keep `train_slice_probes.py` as a standard diagnostic: run it on the first
  CWQ checkpoints to verify the extra data actually moves the train-fit off
  ~97.

**Keep/remove decision (agreed 2026-07-12):** keep the weight-decay fix and
the `lora_dropout` knob; REMOVE the five model-side mechanisms
(`magnetic_eigvec_dropout`, `magnetic_eigvec_shared_mask` + GraphContext
threading, `magnetic_mlp_dropout`, `bias_droppath`, `bias_dropout`) and
`reg_onset_frac`/`RegOnsetCallback`, with their flags and tests. Sequence:
finish round-2 train-slice → commit the experiment exactly as run + tag
(`reg-probes-2026-07`) → removal commit on top. TODO_reg.md and configs
015–017 stay in-tree as the record; reproducing the removed arms = checkout
the tag.

## Motivation

The train-slice diagnostic (job 110782, 200 sampled train questions) showed both
arms in the memorization regime, with the graph arm fitting train *better* than
flat (96.7 vs 95.4 F1; 100% Hits@1) while generalizing *worse* (78.5 vs 80.0
answerable-only test F1). Code inspection confirmed the graph-bias pathway is the
least-regularized part of the model — literally zero regularization:

1. **No dropout anywhere in `src/models/bias.py`** — neither inside the
   MagneticBias MLPs nor on the summed bias tensor. LoRA dropout (0.05) acts only
   on backbone adapter inputs; Llama's `attention_dropout` is 0.0.
2. **No weight decay on any graph-bias parameter — unintentionally.**
   `GraphTrainerV2.create_optimizer` (`src/utils/text_graph_trainer_v2.py:60-82`)
   uses HF's `get_decay_parameter_names`, which excludes any param whose *name
   contains the substring "bias"*. Every graph-bias param lives under modules
   named `graph_bias` / `shared_graph_bias` / `bias_modules` → all land in the
   `bias_no_decay` group. The `bias_decay` group (line 81) is dead code today,
   despite `weight_decay=0.1` in TrainingArguments.

MagneticBias is the prime memorization suspect: a deep-set MLP over 128
magnetic-Laplacian eigenvectors — close to a fingerprint of each question's
subgraph. SPDBias (64×H lookup table, shared across all graphs) cannot memorize
individual examples.

Expectation-setting: these probes may recover part of the 1.5-pt graph-vs-flat
deficit and de-risk the D5 scale recipe; none will close the 18-pt
generalization gap (that needs the CWQ data arm). The most valuable outcome is
diagnostic — see Part 7.

---

## Part 1 — Weight-decay fix (`src/utils/text_graph_trainer_v2.py`)

**Change** in the grouping loop (lines 68–74), split the decay decision by group:

- **Base params (backbone/LoRA): keep HF behavior exactly** — `n in
  decay_parameters` as today. LoRA A/B matrices already get decay; unchanged.
- **Graph-bias params (`is_active`): decide by shape + explicit opt-out:**
  `has_decay = p.ndim >= 2 and not getattr(p, "_no_weight_decay", False)`.
- **New knob `bias_weight_decay`** threaded into the trainer alongside `bias_lr`
  (constructor arg, same pattern): the decay value for the `bias_decay` group
  instead of `self.args.weight_decay`.
- Add `bias_weight_decay` to `save_run_metadata` extras and the trainer's wandb
  log lines so run records carry it.

**Decay policy (DECIDED 2026-07-11: "exempt SPD table"):**

| parameter | shape (1B) | semantic role | decay? |
|---|---|---|---|
| `SPDBias.weights` | (64, 32) | additive logit lookup table | **NO** — tag `_no_weight_decay = True` in `SPDBias.__init__` |
| `lambda_lin.weight` | (64, 1) | eigenvalue → feature transform | yes |
| `deep_set.0.weight` | (128, 128) | feature matrix | yes |
| `proj.0.weight` | (128, 256) | feature matrix | yes |
| `proj.2.weight` | (32, 128) | output head (zero-init) — the suspected fingerprint module; shrinkage-toward-no-magnetic-bias is the point | yes |
| all `.bias` vectors, 1-D gains (Laplacian/RWSE) | — | additive / per-head gains | no (caught by `ndim >= 2` rule) |

Rationale for the SPD exemption: the table is 2-D by shape but semantically an
additive attention-logit bias per bucketed distance — it transforms nothing, and
with 64 globally-shared values per head it *cannot* memorize examples. Decay on
it would fight legitimately general signal for zero anti-memorization benefit.
RRWP's MLP (not used in KGQA) gets the same treatment as Magnetic's by the rule.

Mechanism keeps the trainer generic: bias modules tag exempt params locally with
the standard `param._no_weight_decay = True` attribute; the trainer never
imports bias types.

**Default = `bias_weight_decay: 0.0` — preserves today's accidental behavior
bit-for-bit.** The probe arm sets 0.1 explicitly. Rationale: keeps every
existing config/run comparable (repo convention: pinned control arms). Flip the
default only after a probe validates it.

AdamW note: steady-state shrinkage is governed by `wd` itself (Adam-normalized
gradients balance `wd·w` independent of lr), so `bias_lr=5e-3` doesn't amplify
`wd=0.1` — it only reaches equilibrium faster. 0.1 is a sane single probe value.

## Part 2 — Bias-path dropout (`src/models/bias.py` + `src/models/config.py`)

Three new `GraphConfigMixin` fields (all default **0.0 = exactly current
behavior**; stored flat → free `save_pretrained` / `graph_bias_config.json`
round-trip; old checkpoints load via existing `getattr(..., default)` pattern):

| field | acts on | mechanism |
|---|---|---|
| `magnetic_eigvec_dropout` | `MagneticBias.forward` | per-forward keep-mask over the M eigenvector axis |
| `magnetic_mlp_dropout` | `MagneticBias` MLPs | functional dropout after each SiLU |
| `bias_droppath` | `GraphAttentionBias.forward` | per-sample whole-bias drop, survivor rescaling |

**Eigenvector dropout mechanics:** when `self.training and p > 0`, sample
`keep ~ Bernoulli(1-p)` of shape `(B, M)`, then (a) AND into `valid` so dropped
eigenvectors leave `h_avg`, (b) zero the corresponding columns of
`V_real`/`V_imag` *before* the einsums (they contract over eigenvector index
`l`, so zeroed columns contribute exactly nothing — do it upstream so folded and
`legacy_unfolded` branches share one code path), (c) rescale the einsum output
by `1/(1-p)` so expected bias magnitude matches eval. No phase/sign
augmentation — the `V·V†` products are already phase-invariant (a no-op there).

**MLP dropout:** apply **functionally** (`F.dropout(..., training=self.training)`)
after `deep_set`'s SiLU and after `proj[1]`'s SiLU — do NOT insert `nn.Dropout`
into the Sequentials, which would shift child indices and break existing
checkpoint param names (`proj.2.weight`), the `proj[2]` references
(bias.py:123-125, 153-155, 193, 203), and `select_active_params` substring
matching.

**Whole-bias DropPath:** at the end of `GraphAttentionBias.forward`, after
summing modules but **before** the K-hop gate (the gate is a hard mask, not
signal — dropping it would change which positions are attendable): per-sample
`(B,1,1,1)` Bernoulli, zero dropped samples' bias, scale survivors by
`1/(1-p)`. Per-layer independent masks (called once per layer). Frozen recipe
runs `k_hop=0`, but the ordering keeps the knob safe for k-hop runs.

## Part 3 — LoRA dropout knob (`kgqa/config.py`)

`lora_dropout: float = 0.05` on `RunConfig`, consumed by `lora_config()`
(replacing the hard-coded 0.05), plus `--lora-dropout` flag. Default = current
value. Verify `flat_train.py` reads `lora_config()` so the knob is honored in
both arms.

## Part 4 — KGQA plumbing (`kgqa/config.py`, `__main__.py`, `train.py`)

- New `RunConfig` fields (train-key group): `bias_weight_decay=0.0`,
  `magnetic_eigvec_dropout=0.0`, `magnetic_mlp_dropout=0.0`, `bias_droppath=0.0`
  (+ `lora_dropout=0.05` from Part 3).
- `bias_params()` additionally emits the three model-side knobs **only when
  nonzero** (matches its "only enabled features contribute keys" convention) →
  flow into `GTLMLlamaConfig` at `train.py:115` with zero further wiring.
- **Deliberately NOT in `data_config_key()`** — pure training knobs, no dataset
  rebuild; all probes reuse the existing nmax50 `.gtds` cache.
- `train.py`: pass `bias_weight_decay=cfg.bias_weight_decay` to
  `KGQAGraphTrainer`; add new knobs to `save_run_metadata` extras.
- `__main__.py`: five new flags; `validate()` gains range checks
  (`0 ≤ p < 1` for the three dropout rates, `bias_weight_decay ≥ 0`).

## Subtleties verified (no action needed, load-bearing)

1. **Gradient checkpointing:** per-layer bias recomputed inside
   `torch.utils.checkpoint.checkpoint(_bias, use_reentrant=False)`
   (`dispatch.py:236`) — `preserve_rng_state=True` default replays dropout masks
   identically in recompute → correct gradients. Update the "Safe to recompute —
   deterministic" docstring to note the reliance on `preserve_rng_state`.
2. **Eval-time bias cache** (`bias.py:318, 336`): reads/writes gated on
   `not self.training`, dropout training-only — no stale-mask hazard.
3. **Flex path:** dropout changes tensor values, never shapes — no new dynamo
   recompiles.
4. **`magnetic_shared`:** computed once per forward outside checkpointed layers
   → one dropout mask shared by all layers per step. Acceptable; KGQA uses
   per-layer `magnetic`. Document as a footnote.

## Part 5 — Tests (extend `tests/test_graph_bias.py` + one trainer test)

1. **Decay grouping:** tiny GTLM model → `create_optimizer` with
   `bias_weight_decay=0.1`; assert magnetic matrices (`lambda_lin.weight`,
   `deep_set.0.weight`, `proj.0.weight`, `proj.2.weight`) in the 0.1-decay
   group; `SPDBias.weights` + all `.bias` vectors in decay-0 groups;
   backbone/LoRA groups byte-identical to HF's current split. Assert
   `bias_weight_decay=0.0` default reproduces today's grouping exactly
   (no-regression guarantee).
2. **Eigenvector dropout:** eval mode ⇒ exactly p=0 output; train mode, seeded
   RNG ⇒ dropped eigenvector contributes nothing (compare vs manually truncated
   V); mean over many draws ≈ p=0 output (rescaling correctness).
3. **DropPath:** per-sample all-or-nothing (dropped sample's slice exactly 0,
   kept slice exactly `bias/(1-p)`); K-hop gate `-inf` positions unaffected.
4. **Checkpointing+dropout:** CPU (pattern from `test_flex_cpu.py`) — grads with
   `checkpoint_graph_bias=True` match `False` under identical RNG seed.
5. **Flags:** `test_kgqa_flags.py`-style round-trip for the five new flags +
   `validate()` rejections.

## Part 6 — Probe sweep: `configs/015_regularization_probes.jsonc`

Base = frozen recipe from `013_frozen_nmax50.jsonc` verbatim (r64, lr 1e-4,
bias_lr 5e-3, k_hop 0, nmax50, 15 epochs, dfv3, flex/bf16, B300 partition,
existing data cache). One bundle axis `arm`, **not crossed** (direction-finding
round; combine winners in round 2 if warranted):

| arm | knobs | seeds | hypothesis tested |
|---|---|---|---|
| `wd` | `bias_weight_decay=0.1` | 3 | undecayed bias channel drives memorization |
| `eig` | `magnetic_eigvec_dropout=0.1`, `magnetic_mlp_dropout=0.1` | 3 | magnetic spectrum acts as example fingerprint |
| `droppath` | `bias_droppath=0.1` | 3 | model leans exclusively on structural channel |
| `lora_do` | `lora_dropout=0.15` | 2 | generic overfit lives in LoRA, not graph channel |

The two magnetic dropouts are bundled intentionally (same hypothesis; splitting
costs 3 runs for a distinction that matters only if the bundle works).

**Control = the three existing frozen_nmax50 seeds** — every default preserves
current behavior exactly, so no control re-runs. 11 runs ≈ 22–33 GPU-h,
`max_concurrent: 3-4` on B300.

## Part 7 — Readout

Per arm vs frozen-recipe control:
(a) dev/test F1 + Hits@1 from standard run records;
(b) **rerun `results/error_analysis/train_slice_eval.py` on each arm's best
checkpoint** — same 200 train questions — the decisive readout is the
*train-fit vs test gap*, not test F1 alone.

Interpretation grid:
- `eig`/`droppath` shrink train-slice F1 from ~96.7 toward ~90 while test F1
  holds or rises → structural channel confirmed as the memorization vector;
  D5 scale recipe carries the knob.
- Train-fit stays ~97, test moves nowhere → graph channel exonerated;
  memorization lives in LoRA; CWQ data arm is the only real lever.

Summary table + verdict → the sweep's results dir.

## Execution order

Parts 1–5 (code + tests, run suite) → Part 6 sweep config → launch on
instruction → Part 7 analysis.
