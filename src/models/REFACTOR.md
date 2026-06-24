# GTLM refactor plan — making the model backbone-agnostic

Status: **implemented (Llama, Strategy B).** Steps 1–4 of §5 are done: the flat
package layout, the split into `structural_mask.py` + `dispatch.py` with the
registered `gtlm_*` functions, the extracted mixins + typed `GraphContext`, and
the thin `modeling_gtlm_llama.py` adapter. The full v1↔v2 parity suite is green
against the Strategy-B model. This document remains the design record; the
§3c contract describes how to add the next backbone.

Note: `register_for_auto_class` (Hub `trust_remote_code` source-bundling) **is
enabled** — `save_pretrained` bundles the adapter and its flat module closure
into the checkpoint and sets `auto_map`, so
`AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)` works for
Hub-shared checkpoints without the package installed. Two constraints from HF's
bundler shape the layout (verified end-to-end against transformers 4.50.3):

- **The closure must be flat with single-dot imports.** HF's source-bundler
  regex only follows `from .<mod> import …` (single-dot, same directory). It
  cannot resolve `..` parent-package imports or subpackages, and it does **not**
  match the `from . import <mod>` form (the dot must be glued to the module
  name). So every GTLM module lives flat in `src/models/` and `dispatch.py`
  imports flex_kernel as `from .flex_kernel import …`, not `from . import
  flex_kernel`.
- **Local-checkpoint loads copy only the entry file's *direct* imports.** When a
  checkpoint is loaded from a local directory, `get_cached_module_file` does not
  recurse into transitive imports (Hub *downloads* do). So
  `modeling_gtlm_llama.py` carries an explicit "bundling manifest" — one
  `from .<mod> import name` per flat module not already pulled in directly —
  pinning the whole closure for local loads too. The in-process
  `AutoConfig`/`AutoModelForCausalLM.register` calls remain for the
  package-installed path.

Goal: adapting GTLM to a new base LLM (Qwen2, Mistral, Gemma2, …) should be a
~40–80 line adapter file, not a 1,100-line fork. Everything that is *not*
intrinsically tied to a specific backbone should be written once.

---

## 1. Current state

```
src/models/
  modeling_gtlm_llama_v2.py   # GTLMLlama{Config,Attention,DecoderLayer,Model,ForCausalLM}
  graph_attention_v2.py       # structural mask, node→token expand, dispatch, flex seams
  graph_bias.py               # BaseBias registry + GraphAttentionBias
  flex_kernel.py              # FlexAttention BlockMask / score_mod / compiled call
  model_utils.py              # bias-parameter save/load
  configs/                    # json configs
  flex_attn/                  # benchmark suite (dev-only)
  legacy/                     # v0, v1  ← just moved here
```

The classes are already named without a `_v2` suffix (`GTLMLlamaForCausalLM`,
etc.); only the **file** carries `_v2`. So the rename is about modules/paths, not
public class names.

---

## 2. Audit — what is model-agnostic vs. model-specific

This is the core of the plan: I went through v2 line by line, including
*implicit* Llama coupling (HF APIs that are not named "llama" but are
Llama-shaped). "Model-specific" below means *would have to change or could
silently break when the backbone changes* — not just "mentions Llama".

### 2a. Already fully model-agnostic (reuse verbatim)

| File / piece | Why it's agnostic |
|---|---|
| `graph_bias.py` (all of it) | Pure `nn.Module` + tensor ops. No transformers import at all. |
| `flex_kernel.py` (all of it) | Pure `torch` FlexAttention. No backbone coupling. |
| `model_utils.py` | `state_dict` save/load by name-substring. Backbone-neutral (but fragile — see §6). |
| `graph_attention_v2.build_dense_structural_mask` | Tensor ops over node_ids/prompt/pad/k-hop. |
| `graph_attention_v2.expand_node_to_token_bias` | Pure gather. |
| `graph_attention_v2` flex seams (`build_flex_block_mask`, `make_soft_score_mod`, `flex_attention_forward`) | Thin wrappers over `flex_kernel`. |

### 2b. Agnostic *logic* currently welded into Llama subclasses (extract, don't rewrite)

These live inside `GTLMLlama*` classes but contain almost no backbone-specific
code. They should move into shared mixins/functions:

- `GTLMLlamaForCausalLM.forward` orchestration: reset bias cache, build pad mask,
  pick backend, build structural mask **or** flex BlockMask, assemble `ctx`,
  install `ctx` on every attention module, call the inner model, slice logits,
  compute loss. **All backbone-neutral** *given* the adapter contract in §3c.
- `generate` override (resets the per-generation bias cache).
- `prepare_inputs_for_generation` — the graph half (extend `node_ids` for new
  tokens) is agnostic; the KV-cache slicing half is copied HF boilerplate.
- `from_pretrained` — the 3-scenario loader (LoRA / bias-only / standard) is
  agnostic; it just calls `super().from_pretrained`.
- `_sanitize_attn_config` — FA2 downgrade, impl validation, pinning
  `_attn_implementation`. Agnostic.
- Config graph fields + auto-class registration — agnostic pattern.

### 2c. Genuinely model-specific (must be provided per backbone)

| # | Coupling | Where (v2) | Notes / failure mode |
|---|---|---|---|
| 1 | **q/k/v/o projection + RoPE + KV-cache update** copied from `LlamaAttention.forward` | `modeling_gtlm_llama_v2.py:172-189, 248-249` | `apply_rotary_pos_emb` is Llama's RoPE. Qwen2 matches; **Gemma/Phi/GPT-NeoX use partial/different RoPE**. This block is the bulk of the per-backbone rewrite. |
| 2 | **`head_dim` computed as `hidden_size // num_attention_heads`** | `:148-149` | **Latent bug.** `config.head_dim` exists in HF 4.50 (defaulted to 128 for base Llama) and is set *explicitly* on models where it ≠ hidden/heads (Gemma2=256). Must read `config.head_dim`. |
| 3 | **Attention `forward` signature must match what the backbone's `DecoderLayer.forward` passes** | inherited `LlamaDecoderLayer.forward` calls `self_attn(hidden, position_embeddings, attention_mask, past_key_value, cache_position, **kwargs)` | Other backbones pass different positional/kwarg shapes (e.g. `position_ids` vs `position_embeddings`). This is *why* a per-backbone attention class is unavoidable. |
| 4 | **"Separate q/k/v/o proj" layout** assumption | `:175-177, 249` | GPT-NeoX/Falcon fuse QKV (`query_key_value`). Such backbones need a different projection block. |
| 5 | **Attention features the GTLM path cannot express** | — | **Attn-logit softcapping (Gemma2), sliding-window (Mistral/Gemma2), QK-norm (Qwen3), partial rotary (Phi)**. Our dispatch replaces the score computation, so any of these applied *inside* the stock attention must be re-added or declared unsupported. |
| 6 | **Submodule wiring for ctx install + heads** | `:336, 422-423, 438-440` | Relies on `model.embed_tokens`, `model.layers[i].self_attn`, `lm_head`. HF-conventional for most decoder LMs but not universal (GPT-2 = `transformer.h[i].attn`). Should go behind overridable accessors. |
| 7 | **`eager_attention_forward` imported from the llama module** | `graph_attention_v2.py:36-39` | Functionally generic, cosmetically Llama-located. Re-home to a neutral import (or capture per backbone). |
| 8 | **Copied HF internals drift** | attention forward (#1), prepare_inputs cache slicing | These are snapshots of HF code; a transformers bump can desync them silently. Strategy choice in §3a directly mitigates this. |

---

## 3. Proposed architecture

### 3a. The central decision: how the graph bias enters attention

**Strategy A — override the attention `forward` (what v2 does today).**
Each backbone's attention subclass copies that backbone's projection + RoPE +
cache code and splices the graph dispatch into the middle.
- Pros: explicit; works on any HF version; full control.
- Cons: copies fragile internals (#1, #8); must re-handle RoPE/QK-norm/softcapping
  per backbone (#5); the most code per backbone.

**Strategy B — register a custom attention *function* (HF 4.50 supports this).**
HF dispatches attention through `ALL_ATTENTION_FUNCTIONS[config._attn_implementation]`
with the generic signature `(module, query, key, value, attention_mask, scaling,
dropout, **kwargs)`, called *after* the backbone's own q/k/v projection + RoPE +
QK-norm + cache update. We register `gtlm_eager` / `gtlm_flex`
functions (written **once**, agnostic) that read the per-layer `graph_bias`
and the per-batch `ctx` off `module`, and apply the structural mask + soft bias.
(A third `gtlm_sdpa` backend existed in the original refactor but was later
removed — see the §7 note — so the surviving dense backend is `gtlm_eager`.)
The backbone's *stock* attention `forward` is left untouched.
- Per-backbone code collapses to: an attention subclass whose `__init__` adds
  `self.graph_bias` (**no `forward` override**), the decoder/model swaps, and the
  config/causal-LM mixin bindings. RoPE, QK-norm, partial rotary, cache, and the
  q/k/v/o layout all come from the backbone for free (#1–#4 mostly evaporate).
- Cons: depends on the HF attention-interface contract (present in 4.50, but
  version-sensitive — #8 shifts from "copied code drifts" to "interface contract
  changes"); softcapping/sliding-window done *inside* the stock fn (#5) must be
  replicated in our `gtlm_*` fns where present; custom impl name must pass HF's
  `_attn_implementation` validation (register before model init).

**DECIDED: Strategy B.** It delivers "new backbone in ~40 lines" and removes the
largest fragility (copied forwards). A is kept only as a documented fallback for
a hypothetical backbone whose attention forward does not route through
`ALL_ATTENTION_FUNCTIONS`.

> **Spike confirmed (2026-06-18, HF 4.50.3, CPU).** Registered a custom fn, set a
> tiny `LlamaForCausalLM` to use it, and verified the *stock* forward called it
> once per layer with post-RoPE q/k/v `(B,H,T,head_dim)`, with `module._graph_ctx`
> readable inside. Two implementation notes:
> - `ALL_ATTENTION_FUNCTIONS` is a plain `dict` in 4.50.3 → register via
>   `ALL_ATTENTION_FUNCTIONS["gtlm_eager"] = fn` at import time. (The
>   `AttentionInterface.register` class API does not exist in this version.)
> - Set `config._attn_implementation` **directly** (as v2 already does to pin
>   `"eager"`); do *not* route a custom name through the public
>   `attn_implementation=` kwarg / `from_pretrained`, which validates against the
>   built-in set and would reject it.

#### 3a-bis. Where the graph-bias parameters live, init, and saving

Approach B makes the *attention function* stateless and shared; the trainable
parameters still live on a real submodule, so they keep first-class autograd /
`state_dict` / optimizer / device-dtype / gradient-checkpoint behavior.

- **Init location (unchanged from v2):** the per-backbone attention subclass adds,
  in `__init__`, `self.graph_bias = GraphAttentionBias(num_heads, head_dim=config.head_dim, layer_idx, bias_config=config)`.
  One per layer, instantiating only the *enabled* bias types. Parameters live at
  `model.layers.{i}.self_attn.graph_bias.bias_modules.{j}.*`.
- **The `gtlm_*` function is stateless:** HF passes it the attention `module`; it
  *reads* `module.graph_bias` and `module._graph_ctx` and owns nothing. (The flex
  `score_mod` captures the computed `(B,H,N,N)` bias *activation*, not the params.)
- **Zero-init preserved:** the `_is_hf_initialized = True` guards on the
  zero-initialised output layers keep GTLM ≡ base model at step 0 (bias trains up
  from zero); HF `post_init` won't clobber them.
- **Saving (redesigned — D2 frees us from the old format):**
  1. *Full model* — standard `save_pretrained`/`from_pretrained`; bias params are
     ordinary submodule params, included automatically.
  2. *Bias-only / LoRA* (frozen-backbone training; don't write the multi-GB base)
     — select the bias params by **module type** (`isinstance(m, GraphAttentionBias)`),
     not by name substring as today's `model_utils.py` does. Type-based selection
     can't silently miss/over-match and survives renames. Becomes `io.py`;
     load with `load_state_dict(strict=False)`.

### 3b. Package layout (proposed)

GTLM is the only model in this repo, so `src/models/` *is* the GTLM package. The
shared library modules and the per-backbone adapters all sit **flat** in
`src/models/` — no `backbones/` subpackage. This flatness is not just taste: HF's
`trust_remote_code` source-bundler can only follow single-dot, same-directory
imports (see the status note up top), so a subpackage would break Hub loading.
Each backbone adapter is a `modeling_gtlm_<backbone>.py` file; the
`modeling_gtlm_` prefix makes adapters cluster in folder listings and signals "HF
entry file with the core class" the way `modeling_*.py` does on the Hub. The
library modules coexist with the (visually distinct) support dirs `legacy/`,
`configs/`, and the `flex_attn/` benchmark suite.

```
src/models/
  __init__.py                # re-export GTLMLlama* (re-exports the adapter; registration is a side-effect of the adapter import)
  modeling_gtlm_llama.py     # GTLMLlama{Config,Attention,DecoderLayer,Model,ForCausalLM} + registration + bundling manifest
  # modeling_gtlm_qwen2.py, modeling_gtlm_mistral.py, … added later, each ~40–80 lines
  bias.py                    # ← graph_bias.py (verbatim)
  flex_kernel.py             # ← flex_kernel.py (verbatim, stays in place)
  structural_mask.py         # ← build_dense_structural_mask + expand_node_to_token_bias
  dispatch.py                # backend dispatch + the gtlm_eager/flex attn functions
  context.py                 # the per-batch GraphContext (typed) + install/reset helpers
  config.py                  # GraphConfigMixin (the flat graph fields + validation)
  causal_lm.py               # GraphCausalLMMixin (forward/generate/prepare/from_pretrained)
  attention.py               # GraphAttentionMixin (owns graph_bias; Strategy-A helper too)
  io.py                      # ← model_utils.py (bias param save/load)
  README.md                  # current README.md (user manual), paths updated
  REFACTOR.md                # this document
  legacy/                    # v0, v1 (already moved)
  configs/
  flex_attn/
```

Most moves stay *in place* in `src/models/` (rename only): `graph_bias.py → bias.py`,
`model_utils.py → io.py`, `flex_kernel.py` unchanged. `graph_attention_v2.py` splits
into `structural_mask.py` + `dispatch.py` because it currently mixes the two
concerns; the flex seams fold into `dispatch.py`. The adapter
`modeling_gtlm_llama_v2.py` becomes `modeling_gtlm_llama.py`. No new dirs.

Registration side-effects fire on import: `dispatch.py` registers the `gtlm_*`
functions into `ALL_ATTENTION_FUNCTIONS`, and `modeling_gtlm_llama.py` does the
`AutoConfig`/`AutoModel` + `register_for_auto_class` registration. `__init__.py`
re-exports the adapter, so importing anything under `src.models` triggers all of
it. Trade-off accepted (rather than scoping to a `gtlm` import). Fine for a
single-model repo.

### 3c. The backbone adapter contract

A backbone adapter must supply (most are one-liners or class attributes):

1. **Config**: `class GTLMXConfig(GraphConfigMixin, XConfig)` with a stable
   `model_type`.
2. **Attention**: subclass of the backbone's attention that, in `__init__`, adds
   `self.graph_bias = GraphAttentionBias(num_heads, head_dim=config.head_dim, …)`
   and **no `forward`** (Strategy B). It also sets `config._attn_implementation`
   to the matching `gtlm_*` function name.
3. **Decoder layer / model**: swap in the GTLM attention / GTLM layers (init-only).
4. **Causal LM**: `class GTLMXForCausalLM(GraphCausalLMMixin, XForCausalLM)`.
5. **Accessors** (only if the backbone deviates from convention): override
   `_iter_attn_modules(self)` / `_layers(self)` / lm-head/embedding hooks so the
   mixin's ctx-install and dtype probing stay backbone-neutral (#6).
6. **Capability declaration**: a small set, e.g.
   `GTLM_SUPPORTED = {...}` / `GTLM_UNSUPPORTED = {"sliding_window", "attn_logit_softcapping"}`,
   so unsupported features raise a clear error instead of silently mis-computing
   (#5).

Registration (`AutoConfig.register` + `AutoModelForCausalLM.register` +
`register_for_auto_class`) lives at the bottom of each `modeling_gtlm_<backbone>.py`,
next to a **bundling manifest** — explicit `from .<mod> import name` lines pinning
every flat module so local `trust_remote_code` checkpoint loads copy the whole
closure (see status note up top).

---

## 4. Naming scheme (dropping `_v2`)

- **Files/modules**: `modeling_gtlm_llama_v2.py` → `modeling_gtlm_llama.py`;
  `graph_bias.py` → `bias.py`; `graph_attention_v2.py` →
  `structural_mask.py` + `dispatch.py`; `flex_kernel.py` unchanged;
  `model_utils.py` → `io.py`. All flat in `src/models/`. New backbones add a
  `modeling_gtlm_<backbone>.py` next to the Llama one.
- **Public class names**: unchanged — `GTLMLlamaConfig`, `GTLMLlamaForCausalLM`,
  `GTLMLlamaAttention`, etc. New backbones follow `GTLM<Backbone>...`.
- **`model_type`**: keep `"gtlm_llama"` (sensible name; no longer a back-compat
  constraint — see D2). New backbones get `"gtlm_qwen2"`, etc.
- **Convenience re-exports** in `src/models/__init__.py` so users write
  `from src.models import GTLMLlamaForCausalLM`.

---

## 5. Migration steps

1. Rename in place (git mv) within `src/models/`: `graph_bias.py → bias.py`,
   `model_utils.py → io.py`; `flex_kernel.py` stays. Add `src/models/__init__.py`
   as the package entry point. Import-path fixes only.
2. Split `graph_attention_v2.py` → `structural_mask.py` + `dispatch.py`; add the
   `gtlm_eager/sdpa/flex` attention functions and register them (Strategy B).
3. Extract `GraphConfigMixin`, `GraphCausalLMMixin`, `GraphAttentionMixin`,
   `GraphContext` from `modeling_gtlm_llama_v2.py`.
4. Write `modeling_gtlm_llama.py` as the thin adapter; confirm it reproduces v2.
5. **Clean break (D4):** update every consumer (~6 src files + tests + user-manual
   README) to the new import paths. No compat shim left behind.
6. Run the existing v2 test-suite (parity vs v1, sdpa==eager, k-hop, flex,
   bucketing) unchanged — it is the correctness oracle for the refactor.

Scope (D3): **Llama + seams only** — no second backbone in this pass; the contract
in §3c documents how one is added later. Each step keeps the test-suite green;
the refactor is behavior-preserving by construction (v1 remains the numerical
reference in `legacy/`).

---

## 6. Secondary cleanups (all in scope)

- Read `config.head_dim` everywhere (fixes #2). **In scope.**
- Dead code: the `require_*` properties in `graph_bias.py` are unused by v2
  (only v1 used them); drop. **In scope.**
- Replace `model_utils.py` substring save/load with type-based selection in
  `io.py` (see §3a-bis). **In scope.**
- `graph_attention_dispatch`'s `impl="flex"` branch only raises; it disappears
  with the dispatch split. **In scope.**

---

## 7. Decisions (locked 2026-06-18)

- **D1 — Attention strategy: B.** Register `gtlm_eager/flex` functions into
  HF's `ALL_ATTENTION_FUNCTIONS`; backbones keep their own attention forward. See
  §3a / §3a-bis.
  > **Update (post-refactor): `gtlm_sdpa` removed.** The original design shipped a
  > third `gtlm_sdpa` backend, but a custom *dense* attention bias makes the fused
  > SDPA kernels either ineligible (flash) or both buggy in backward (mem-efficient,
  > GQA + per-head bias → `LSE is not correctly aligned`) and pointless (the dense
  > bias is already materialized). SDPA therefore only ever reduced to eager, so it
  > was dropped: the two backends are now **eager** (dense) and **flex** (sparse,
  > the actual speedup). The flex incremental-decode fallback uses eager.
- **D2 — Checkpoint back-compat: none.** Use the cleanest implementation; old
  checkpoints are served by `legacy/` if ever needed. This frees the save format
  (type-based bias-only save) and `model_type`.
- **D3 — Scope: Llama + seams only.** No second backbone this pass; the §3c
  contract documents how to add one.
- **D4 — Import paths: clean break.** Update all consumers; no compat shim.
- **D5 — Unsupported attention features: hard-error via the capability
  declaration.** Backbones whose attention applies *score-step* features inside
  the stock fn (Gemma2 logit-softcapping, sliding-window) or score-level sequence
  PEs (ALiBi/T5 — see §8) must either re-implement them in a `gtlm_*` variant or
  declare them unsupported, which raises a clear error. RoPE/QK-norm/partial-rotary
  are stage [2] (before the fn) and need no special handling under B.

---

## 8. Invariants & positional-encoding contract

### 8a. Invariants preserved across the refactor (acceptance criteria)

The refactor is behavior-preserving; these properties must still hold (the v2
test-suite checks all of them):

- **Per-node position reset.** Each node's tokens are RoPE'd as positions 0,1,2,…
  independently (driven by the collator's `position_ids` + the backbone's
  model-level rotary embedding — *not* the attention function), giving prefix
  permutation-invariance. Under B this is *more* faithful: the backbone applies
  its own PE to our `position_ids`; we never re-copy the RoPE call.
- **Padding is loss-neutral** (masked from attention and loss; verified to fp64).
- **Prompt node is packed last** and generated causally; all other nodes attend
  bidirectionally.
- **`graph_attn_impl` backends agree** (flex == eager/dense to tolerance; eager
  matches the original v0 model forward *and* backward).

Three independent channels, none touched by the attention-function swap:
`position_ids` (per-node reset → PE only); `node_ids` (token→node → bias gather +
structural mask); absolute sequence index (`arange` over `kv_len` → causal order).

### 8b. PE-compatibility contract (which backbones compose cleanly)

GTLM's graph bias is itself a *relative, score-level* PE keyed on graph structure.
It composes cleanly with sequence PEs on a **different axis** (intra-node text
order) and conflicts with sequence PEs that occupy the **same axis** it does
(inter-token score-level distance). The per-node position reset is what keeps the
first group on its own axis.

| Sequence PE | Backbones | Injection stage | B handles it? | Composes with graph PE? |
|---|---|---|---|---|
| **RoPE** | Llama, Qwen, Mistral, Gemma, Phi | [2] Q/K rotate | free (backbone owns it) | **Yes, cleanly** — per-node reset neutralizes cross-node distance |
| **Absolute learned/sinusoidal** | GPT-2, OPT | [0] embeddings | free (pre-attention, uses `position_ids`) | **Yes** — intra-node axis only, with per-node-reset `position_ids` |
| **ALiBi** | BLOOM, MPT | [4] additive distance bias | needs a `gtlm_*` variant (same stage as our bias) | **Conflicts across nodes** — raw sequence distance fights the graph bias; needs a per-node-reset ALiBi or declare unsupported |
| **T5 relative bias** | (enc-dec; rare) | [4] learned relative bias | same as ALiBi | same tension as ALiBi |
| **NoPE** | research | none | free | runs; loses intra-node text order — usually undesirable |

Attention pipeline stages referenced above: [0] embeddings · [1] q/k/v projection
· [2] RoPE · [3] KV-cache update · [4] score step `softmax(QKᵀ·scale + mask)·V`
(the only graph-relevant stage; what the `gtlm_*` functions own) · [5] o_proj.

Practical note: the realistic backbone targets (Llama/Qwen/Mistral/Gemma/Phi) are
all RoPE-family, so B's only real downside (score-step PEs) does not arise unless
ALiBi-style backbones are deliberately targeted.
