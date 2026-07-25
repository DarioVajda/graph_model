# GTLM-Llama — user manual

A graph-biased Llama causal LM. Each attention layer combines a **trainable
soft graph bias** (SPD / Laplacian / RWSE / RRWP / Magnetic) with a **structural
mask** (causal + bidirectional-prefix + K-hop + padding), so the model can read
text attached to graph nodes while respecting graph structure.

The implementation is **backbone-agnostic**: all graph logic lives in shared,
backbone-neutral modules, and each base LLM is a thin `modeling_gtlm_<backbone>.py`
adapter (flat in this package). The **Llama-3** family is the production backbone
(`modeling_gtlm_llama.py` — `GTLMLlamaForCausalLM` / `GTLMLlamaConfig`); **BLOOM**
(ALiBi) and **Gemma-3** (layer-heterogeneous RoPE) are positional-encoding probe
adapters running the same machinery on other PE schemes (see the two sections at the
end); `v0`/`v1` are legacy. Import the public classes from the package:
`from src.models import GTLMLlamaForCausalLM, GTLMLlamaConfig` (see REFACTOR.md §3c
to add a backbone).

Checkpoints are **HuggingFace Hub-compatible**: `save_pretrained` bundles the model
code into the checkpoint, so others can load a published checkpoint with
`AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)` without
installing this package.

The moving parts:

| Component | File | Role |
|---|---|---|
| Backbone adapter | `modeling_gtlm_llama.py` | wires Llama base classes to the graph mixins (`GTLMLlamaForCausalLM` / `GTLMLlamaConfig`); HF auto-class registration + bundling manifest |
| Orchestration mixins | `causal_lm.py` / `config.py` / `attention.py` | backbone-neutral forward / generate / loader, config fields, per-layer bias owner |
| Attention functions | `dispatch.py` | the registered `gtlm_eager`/`gtlm_flex` functions + backend dispatch |
| Structural mask | `structural_mask.py` | shared causal + bidirectional-prefix + K-hop + padding mask, node→token bias |
| Bias modules | `bias.py` | the per-layer `GraphAttentionBias` (SPD/Laplacian/RWSE/RRWP/Magnetic) |
| Flex kernel | `flex_kernel.py` | the FlexAttention BlockMask builder, score-mod gather, compiled forward |
| Dataset | `../utils/text_graph_dataset.py` | `TextGraphDataset` — stores topology + precomputed features |
| Collator | `../utils/text_graph_collator_v2.py` | `GraphCollatorV2` — packs a batch into the model's columns |

---

## Quick start

### 1. Build a dataset and precompute features

```python
from src.utils.text_graph_dataset import TextGraphDataset

# graphs: list of networkx graphs; each node has a 'text' attribute and
# graph.graph['prompt_node'] marks the node to generate.
ds = TextGraphDataset(graphs, rcm_ordering=True)   # RCM recommended for flex
ds.compute_shortest_path_distances()               # SPD bias
ds.compute_magnetic_lap(q=0.25)                    # magnetic bias
ds.tokenize(tokenizer, add_eos=True)               # adds 'input_ids'
ds.compute_labels(get_graph_labels)                # adds 'labels'
ds.save("my_dataset")                              # -> my_dataset.gtds/
```

Compute only the features whose biases you enable in the config (see the bias
table below). `rcm_ordering=True` must come **before** the `compute_*` calls so
features are produced in the reordered layout.

### 2. Collate and train

```python
from transformers import AutoConfig, Trainer, TrainingArguments
from src.utils.text_graph_collator_v2 import GraphCollatorV2
from src.models import GTLMLlamaConfig, GTLMLlamaForCausalLM

cfg = GTLMLlamaConfig(
    **AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B").to_dict(),
    spd=True, max_spd=32, magnetic=True, magnetic_dim=32, magnetic_q=0.25,
    k_hop=2, graph_attn_impl="flex",          # "eager" / "flex"
)
model = GTLMLlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", config=cfg)

collator = GraphCollatorV2(
    tokenizer=tokenizer, k_hop=cfg.k_hop, k_hop_directed=cfg.k_hop_directed,
    pad_to_block=True,                        # REQUIRED for graph_attn_impl="flex"
)

Trainer(model=model, args=TrainingArguments(...),
        train_dataset=ds, data_collator=collator).train()
model.save_pretrained("my_model")
```

Build the collator's `k_hop` / `k_hop_directed` from the config so the K-hop
mask matches what the model expects.

### 3. Load and generate

```python
model = GTLMLlamaForCausalLM.from_pretrained("my_model", graph_attn_impl="eager")
batch = collator([dataset_item])                  # one graph
out = model.generate(**batch, max_new_tokens=64)
```

Generation runs the dense backend automatically for the incremental decode steps
(see *FlexAttention → Decoding*), so both `"flex"` and `"eager"` generate
correctly.

A checkpoint published to the HuggingFace Hub loads with no local install —
`save_pretrained` bundled the model code into it:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("org/my-gtlm", trust_remote_code=True)
```

---

## Configuration (`GTLMLlamaConfig`)

`GTLMLlamaConfig` is a `LlamaConfig` plus flat graph fields, so it round-trips
through `save_pretrained`/`from_pretrained` with no custom code.

### Graph-bias fields

| Field | Default | Meaning | Dataset feature required |
|---|---|---|---|
| `spd` | `False` | shortest-path-distance bias | `compute_shortest_path_distances()` |
| `max_spd` | `32` | SPD clamp | — |
| `laplacian` | `False` | Laplacian-eigenmap bias | `compute_laplacian_coordinates()` |
| `rwse` | `False` | random-walk structural encoding | `compute_rwse()` |
| `rrwp` | `False` | relative random-walk positional | `compute_rrwp()` |
| `max_rw_steps` | `8` | RWSE/RRWP steps | — |
| `magnetic` | `False` | magnetic-Laplacian bias | `compute_magnetic_lap()` |
| `magnetic_dim` | `32` | # magnetic eigenvectors | — |
| `magnetic_q` | `0.25` | magnetic potential | — |

### Structural fields

| Field | Default | Meaning |
|---|---|---|
| `k_hop` | `0` | hard K-hop attention gate (0 = off; tokens whose nodes are >K hops apart can't attend) |
| `k_hop_directed` | `False` | follow edge direction for the K-hop gate |
| `graph_attn_impl` | `"eager"` | attention backend: `"eager"` or `"flex"` |
| `checkpoint_graph_bias` | `True` | recompute the per-layer bias in backward (training-only; large memory saving at high node counts) |

### FlexAttention knobs (only used when `graph_attn_impl="flex"`)

| Field | Default | Meaning |
|---|---|---|
| `flex_compile_mode` | `"max-autotune-no-cudagraphs"` | `torch.compile` mode. Autotune is fastest at runtime but pays a one-time \~320s compile per shape; use `"default"` (\~16s) for quick iteration. |
| `flex_block_size` | `None` | BlockMask block size. `None` uses the K-hop gate (64 when `k_hop>0`, else 128); set an int to override. |
| `flex_cache_size_limit` | `32` | raises `torch._dynamo`'s recompile cap on the flex path (see *Recompiles* below). |

---

## Attention backends

Select with `graph_attn_impl`:

- **`eager`** — dense reference, and the only dense backend. Adds the soft bias
  onto a dense `(B,1,L,L)` structural mask and delegates to HF's
  `eager_attention_forward`. Works on CPU; used for training, generation, and
  eval. (The fused SDPA kernels are intentionally not offered: a custom dense
  bias makes flash ineligible and the mem-efficient kernel both buggy in backward
  and pointless once the bias is materialized, so SDPA would only reduce to this.)
- **`flex`** — `torch.compile` FlexAttention. The bias stays at node level
  `(B,H,N,N)` and is gathered inside the kernel; the structural mask becomes a
  sparse `BlockMask` so fully-masked blocks are skipped. **Much faster and far
  lower memory** (especially with `k_hop>0` where graphs are sparse); needs CUDA and a
  block-aligned batch (see below). Falls back to dense for incremental decode.

GTLM-Llama is **incompatible with FlashAttention-2** (it can't express the graph
bias / K-hop / bidirectional prefix); requesting it downgrades to eager with a
warning.

---

## The data pipeline

### `TextGraphDataset`

Stores raw topology (networkx, in RAM) + precomputed features (Arrow on disk).

- **Construction:** `TextGraphDataset(graphs, rcm_ordering=False)`. Each graph
  needs node `'text'` attributes and `graph.graph['prompt_node']`.
- **Feature methods** (call the ones matching your enabled biases):
  `compute_shortest_path_distances()`, `compute_laplacian_coordinates(embedding_dim)`,
  `compute_rwse(max_rwse_steps)`, `compute_rrwp(max_rrwp_steps)`,
  `compute_magnetic_lap(q, m)`, `tokenize(tokenizer, add_eos=)`,
  `compute_labels(fn)`.
- **Persistence:** `ds.save(path)` → `path.gtds/`; `TextGraphDataset.load(path)`.
- **Splitting/merging:** `ds.select(indices)`, `ds_a + ds_b`.

#### RCM node ordering

Reverse Cuthill–McKee reordering makes each node's K-hop neighbourhood
contiguous in the packed sequence, so FlexAttention can skip far more
`BlockMask` blocks. It does not change model outputs at the prompt (the prompt
node is always packed last and the model is permutation-invariant over the
prefix) — it is a pure throughput optimization.

- **At construction (preferred):** `TextGraphDataset(graphs, rcm_ordering=True)`
  reorders before any feature is computed, so features inherit the order.
- **After construction:** `ds.apply_rcm_ordering()` — must be called **before**
  any node-indexed feature exists (it raises otherwise).
- **Provenance:** `ds.node_ordering` is `"original"` / `"rcm"` / `"mixed"` and is
  **persisted across save/load** and propagated through `select`/merge. Check
  `ds.is_rcm_ordered`. A reloaded dataset always knows its ordering.

### `GraphCollatorV2`

Packs a list of dataset items into the model's batch columns (`input_ids`,
`node_ids`, `attention_mask`, `position_ids`, `labels`, `prompt_node`,
`num_nodes`, and the structural feature tensors). Non-prompt nodes are
concatenated first (in node order), the **prompt node is packed last**, and
`position_ids` reset per node.

```python
GraphCollatorV2(tokenizer=None, pad_token_id=None,
                k_hop=0, k_hop_directed=False, magnetic_m=0,
                pad_to_block=False, block_size=128,
                len_buckets=None, node_buckets=None)
```

- `k_hop` / `k_hop_directed` — emit the `(N,N)` K-hop mask (match the config).
- `magnetic_m` — truncate magnetic eigenvectors to the first `m` columns.
- `pad_to_block`, `len_buckets`, `node_buckets`, `block_size` — flex padding;
  see below. For the dense backends leave `pad_to_block=False`.

---

## Using FlexAttention

Flex is opt-in (`graph_attn_impl="flex"`) and aimed at sparse, K-hop training.

### Requirements

1. **CUDA + `torch.compile`.** Flex does not run on CPU.
2. **Block-aligned, bucketed batches:** construct the collator with
   `pad_to_block=True`. This is mandatory — the kernel hits a >10× slowdown at
   non-aligned lengths, and the model raises a clear error if a flex batch isn't
   aligned.
3. **RCM ordering** (`rcm_ordering=True` on the dataset) — recommended; this is
   where flex's block-skipping win comes from.

### Bucketing and recompiles

A separate flex kernel is compiled for each distinct `(L, N)` shape (`L` =
padded sequence length, `N` = padded node count) — **both** drive recompiles.
`pad_to_block=True` rounds both up to coarse buckets so only a handful of shapes
ever occur:

- **L** → multiples of 512 with 1.5× midpoints (512, 1024, 1536, 2048, …).
- **N** → powers of two floored at 32 (32, 64, 128, 256, …).

Override with `len_buckets` / `node_buckets`, each `None` (default ladder), a
sorted `list[int]` of allowed sizes, or a callable `f(x) -> x`. Every `L` bucket
must be a multiple of `block_size`.

`torch._dynamo`'s recompile cap defaults to **8**; past 8 distinct `(L, N)`
shapes the flex frame silently falls back to slow eager. The model raises the
cap to `config.flex_cache_size_limit` (default 32) on the flex path. Keep the
number of `(L, N)` pairs a run actually hits comfortably below it (≈16 is a good
target — `L` and `N` are correlated, so the real count is small).

The one-time autotune cost (~320s/shape) is cached on disk by inductor and
reused across processes; use `flex_compile_mode="default"` while iterating to
skip it.

### Decoding

Flex serves the **full-sequence** case (training + prefill, `q_len == kv_len`).
Incremental decode steps (`q_len < kv_len`) automatically use the dense `eager`
path — flex gives no benefit at `q_len=1` and this keeps generation identical to
the dense backend. No action needed; `generate()` just works.

### Memory knobs

- `checkpoint_graph_bias=True` (default) — recomputes the per-layer bias in
  backward; large saving at high `N` (turns some OOMs into runnable). Training
  only.
- `model.gradient_checkpointing_enable(use_reentrant=False)` — decoder-layer
  checkpointing for long sequences; nests correctly with the bias checkpoint.

For the full benchmark/optimization study behind these defaults (block size,
compile mode, int32 node ids, checkpointing, roofline), see
[`flex_attn/README.md`](flex_attn/README.md).

---

## Properties & gotchas

- **The prompt node is always packed last** and generated causally; all other
  nodes attend bidirectionally. Reordering the prefix (e.g. via RCM) does not
  change the prompt's logits.
- **Padding is loss-neutral.** Padded tokens are masked from attention and loss
  (`labels=-100`); padded nodes are never gathered. Verified to fp64.
- **Not FlashAttention-2 compatible** — see above.
- **Compile the right features:** enabling a bias without computing its dataset
  feature (or vice-versa) is the most common setup mistake.

---

## ALiBi backbones (BLOOM)

`modeling_gtlm_bloom.py` runs the same graph machinery on a base model whose
positional encoding is **ALiBi**, not RoPE — evidence that the graph bias is not
tied to a positional-encoding family. It is a **probe adapter**: eager only,
single-GPU, small graphs (GraphQA scale), and it is not part of the ablation grid.

```python
from src.models import GTLMBloomConfig, GTLMBloomForCausalLM

cfg = GTLMBloomConfig.from_pretrained("bigscience/bloom-1b1",
                                      spd=True, magnetic=True, graph_attn_impl="eager")
model = GTLMBloomForCausalLM.from_pretrained("bigscience/bloom-1b1", config=cfg)
```

Everything else — collator, dataset, trainer, bias modules, structural mask — is
unchanged, and **no shared GTLM module was modified for it**; the adapter absorbs
both deviations locally:

| Deviation | How the adapter handles it |
|---|---|
| BLOOM's attention forward is hand-written and never dispatches through `ALL_ATTENTION_FUNCTIONS`, so `gtlm_eager` is unreachable (Strategy A, not B) | `GTLMBloomAttention.forward` substitutes the additive-mask argument with `structural_mask + token_soft_bias` and delegates to `super().forward()` — BLOOM already adds that tensor to the scores before the softmax, so no attention math is reimplemented |
| ALiBi takes no `position_ids`; stock BLOOM derives positions from `attention_mask.cumsum(-1)`, i.e. raw serialization order | `GTLMBloomModel.build_alibi_tensor` rebuilds the bias over GTLM's per-node `position_ids` (reusing HF's own pretrained slopes), so node *ordering* stops leaking into attention as a recency prior. `node_position_mode="spd_depth"` flows through unchanged and makes ALiBi graph-distance-aware |

A single-node graph with `arange` positions reproduces stock `BloomForCausalLM`
logits exactly, so the pretrained model is untouched in the degenerate case.

`graph_attn_impl="flex"` is **rejected at construction** rather than silently
ignored: the flex path needs the registered `gtlm_flex` function BLOOM never calls,
and the ALiBi term would have to be folded into the `score_mod`.

Experiment configs: `src/experiments/graphqa/configs/007_bloom_alibi_data_prep.jsonc`
(BLOOM-tokenized cache — `model_name` is part of the cache identity, so the Llama
cache is untouched), then `008_bloom_alibi_canary.jsonc` (one run: does it learn?)
and `009_bloom_alibi.jsonc` (9 tasks x 3 seeds, full graph bias in every run).

---

## Layer-heterogeneous RoPE backbones (Gemma-3)

`modeling_gtlm_gemma3.py` runs the same machinery on a base model where **different
layers read positions at different RoPE frequencies**: on gemma-3-1b, 22 of 26 layers
rotate at a local base (`rope_local_base_freq`, 10k) and the other 4 at a global one
(`rope_theta`, 1M). With BLOOM's ALiBi and Llama's single global RoPE that makes three
positional-encoding regimes on one graph stack.

Unlike the BLOOM probe this is **Strategy B, exactly like Llama** — `Gemma3Attention`
dispatches through `ALL_ATTENTION_FUNCTIONS`, so there is no attention `forward`
override and **both** backends are wired:

```python
from src.models import GTLMGemma3Config, GTLMGemma3ForCausalLM

cfg = GTLMGemma3Config.from_pretrained("google/gemma-3-1b-pt",
                                       spd=True, magnetic=True, graph_attn_impl="flex")
model = GTLMGemma3ForCausalLM.from_pretrained("google/gemma-3-1b-pt", config=cfg)
```

Text-only: the multimodal 4b+ checkpoints nest their text config under `text_config`
and do not load through `GTLMGemma3Config`. **No shared GTLM module was modified**; the
three deviations are absorbed in the adapter:

| Deviation | How the adapter handles it |
|---|---|
| Gemma-3 alternates sliding-window layers (512 tokens on 1b) with full-attention ones, applied by editing the 4-D `attention_mask` | GTLM passes `attention_mask=None` and substitutes its own structural mask, so the band never materializes. Deliberate: the window is indexed by **packed serialization order** (`cache_position`, *not* `position_ids` — per-node reset does not shrink it), so it would hide most of the graph from 5 of every 6 layers. `_update_causal_mask` warns once past the window so the divergence is never silent. GraphQA never reaches it (~90 packed tokens) |
| `Gemma3TextModel` builds a `HybridCache` in eval, whose sliding layers hold only `min(sliding_window, max_cache_len)` keys and return the *last* window — silently truncating the KV against a full-length structural mask | A forward pre-hook installs a plain `DynamicCache` first (what `LlamaModel` does anyway), and `_prepare_cache_for_generation` forces `generate` off `cache_implementation="hybrid"` |
| The shared stack applies neither softcapping site (`gtlm_*` ignore `softcap=`; the mixin calls `lm_head` directly) | Gemma-3 sets both fields to `None`, so both omissions are exact — `_sanitize_attn_config` **raises** if a checkpoint reintroduces either, rather than training a backbone that is quietly not the pretrained one. Gemma-2 hits this by design |

A single-node graph at any length up to `sliding_window` reproduces stock
`Gemma3ForCausalLM` logits exactly — which covers every real GraphQA batch.

Experiment configs: `010_gemma3_dualrope_data_prep.jsonc`, then
`011_gemma3_dualrope_canary.jsonc` and `012_gemma3_dualrope.jsonc`.

---

## Testing

```bash
.venv/bin/python -m pytest tests/ -q
```

- `tests/test_modeling_gtlm_llama_v2.py` — v2-vs-v1 parity, v0 backward-grad
  parity, K-hop.
- `tests/test_gtlm_attn_functions.py` — the Strategy-B `gtlm_eager`/`gtlm_flex`
  functions in isolation: registered in `ALL_ATTENTION_FUNCTIONS` and bit-identical
  to the reference dispatch when driven through `module._graph_ctx`.
- `tests/test_modeling_gtlm_bloom.py` — the ALiBi adapter: stock-BLOOM parity on a
  single-node graph, bias reaches the scores + gets gradients, ALiBi built over
  per-node positions, decode, flex refusal.
- `tests/models/test_modeling_gtlm_gemma3.py` — the dual-RoPE adapter: stock-Gemma-3
  parity within the sliding window (and the deliberate divergence past it), both RoPE
  bases reaching their layers, per-node reset driving the rotation, `DynamicCache`
  instead of `HybridCache`, softcapping refusal, save/load round-trip.
- `tests/models/test_flex_attention_gemma3.py` — flex-vs-eager parity on the Gemma-3
  backbone (forward, bias grads, bias checkpointing, decode fallback). **Requires a
  GPU**; auto-skips otherwise.
- `tests/test_graph_bias.py` — `GraphAttentionBias`, the K-hop collator mask, and
  `expand_node_to_token_bias`.
- `tests/test_collator_bucketing.py` — L/N bucketing + fp64 loss-neutrality.
- `tests/test_dataset_ordering.py` — RCM relabel/consistency + ordering label.
- `tests/test_flex_cpu.py` — flex unit logic + eager permutation invariance (CPU).
- `tests/test_flex_attention.py` — flex-vs-eager parity, checkpointing, decode
  fallback, permutation invariance (**requires a GPU**; auto-skips otherwise).
- `tests/test_model_compatibility.py` — legacy weight-transfer parity against the
  pre-plugin `v0`/`v1` graph models.

---

## Not yet implemented / roadmap

- **Compiled-kernel cache shipping** — bundling the compiled flex kernels with a
  saved model (and reusing them on download) so a matching environment skips the
  autotune. Planned; until then, inductor's on-disk cache already gives
  cross-run reuse on the same machine.
- **Per-KV-head bias** (share the graph bias across GQA groups) and **chunked
  cross-entropy** (the L≈70k LM-head memory wall) — see `flex_attn/README.md`.
