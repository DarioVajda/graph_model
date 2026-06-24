"""
Full-model benchmark: Llama-3.2-1B, three configurations.

Methods
-------
  * ``eager`` — ``GTLMLlamaForCausalLM`` (spd + magnetic, K-hop) on the dense
    ``eager`` backend: dense ``(B,1,L,L)`` structural mask + per-layer
    ``(B,H,L,L)`` token-expanded soft bias.
  * ``flex``    — the same model and weights, but each attention layer is routed
    through ``flex_core`` at *runtime* (monkeypatch): the ``(B,1,L,L)`` mask
    build is turned into a no-op and a sparse ``BlockMask`` is built once per
    batch; the per-layer soft bias stays at node level ``(B,H,N,N)`` and is
    gathered inside the kernel. No edits to the model source.
  * ``flash``   — stock ``LlamaForCausalLM`` with ``flash_attention_2``, plain
    causal, no graph structure. Floor reference (not functionally equivalent).

We time forward-only and forward+backward separately and record peak memory.
Unlike the isolation bench, the soft bias here is computed *inside* the model
from trainable params, so the backward includes the real bias backprop — this is
the true training-step cost.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import sys
import time
from types import MethodType
from typing import Optional

import torch

from src.models.flex_attn.inputs import GraphSpec, make_model_batch
from src.models.flex_attn.density import compute_density
from src.models.flex_attn import flex_core
from src.models.flex_attn.bench_isolation import _cuda_time, _mb, emit_result

BASE_MODEL = "meta-llama/Llama-3.2-1B"


# ── flex routing (runtime monkeypatch) ────────────────────────────────────────

def _flex_attn_forward(self, hidden_states, position_embeddings,
                       attention_mask=None, past_key_value=None,
                       cache_position=None, **kwargs):
    """Drop-in replacement for ``GTLMLlamaAttention.forward`` using flex.

    Mirrors the original up to the node-level ``node_bias``, then — instead of
    expanding to ``(B,H,L,L)`` and calling SDPA — gathers the bias inside a
    flex ``score_mod`` against the per-batch ``BlockMask`` stashed on the module.
    """
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    ctx = self._graph_ctx
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    if past_key_value is not None:
        k, v = past_key_value.update(k, v, self.layer_idx,
                                     {"sin": sin, "cos": cos, "cache_position": cache_position})

    feats = ctx["features"]

    def _bias():
        return self.graph_bias(
            dtype=q.dtype, device=q.device, num_nodes=ctx["num_nodes"],
            spd=feats["spd"], laplacian=feats["laplacian"], rwse=feats["rwse"],
            rrwp=feats["rrwp"], magnetic=feats["magnetic"], k_hop_mask=None,
            cache_dict=ctx["cache"],
        )

    # Mirror the real model's bias checkpointing (#7) under the same gate.
    if (getattr(self.config, "checkpoint_graph_bias", False)
            and self.training and torch.is_grad_enabled()):
        node_bias = torch.utils.checkpoint.checkpoint(_bias, use_reentrant=False)
    else:
        node_bias = _bias()
    score_mod = flex_core.make_score_mod(node_bias, ctx["node_ids"])
    out = flex_core.flex_attention_forward(
        q, k, v, block_mask=self._flex_block_mask, score_mod=score_mod,
        scaling=self.scaling, enable_gqa=True,
    )                                                        # (B, H, L, d)
    # flex returns (B,H,L,d); the model's reshape expects the HF (B,L,H,d) layout.
    out = out.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    return self.o_proj(out), None


def _pad_batch_for_flex(batch: dict, block_size: int = 128) -> dict:
    """Pad sequence tensors to a multiple of ``block_size`` (the flex kernel hits
    a ~14× cliff at non-aligned L). Padded tokens get attention_mask=0 (masked),
    labels=-100 (no loss), and node_ids=prompt_node (harmless)."""
    import torch.nn.functional as F
    L = batch["input_ids"].shape[1]
    Lb = flex_core.align_len(L, block_size)
    if Lb == L:
        return batch
    pad = Lb - L
    out = dict(batch)
    out["input_ids"] = F.pad(batch["input_ids"], (0, pad), value=0)
    out["attention_mask"] = F.pad(batch["attention_mask"], (0, pad), value=0)
    if "position_ids" in batch:
        out["position_ids"] = F.pad(batch["position_ids"], (0, pad), value=0)
    if "labels" in batch:
        out["labels"] = F.pad(batch["labels"], (0, pad), value=-100)
    # node_ids → prompt node per row (broadcast).
    pn = batch["prompt_node"].view(-1, 1)
    out["node_ids"] = torch.cat([batch["node_ids"], pn.expand(-1, pad)], dim=1)
    return out


def _install_flex(model, batch, k_hop: int):
    """Patch the model for the flex path and build the shared per-batch BlockMask.

    ``batch`` must already be block-aligned (see :func:`_pad_batch_for_flex`)."""
    import src.models.causal_lm as mod

    # 1) Neutralize the dense (B,1,L,L) structural-mask build (we use a BlockMask).
    mod.build_dense_structural_mask = lambda **kw: None

    # 2) Build the sparse block mask once for this batch.
    node_ids = batch["node_ids"]
    L = node_ids.shape[1]
    pad_mask = batch.get("attention_mask")
    if pad_mask is None:
        pad_mask = torch.ones_like(node_ids)
    block_mask = flex_core.build_block_mask(
        node_ids, batch["prompt_node"], pad_mask, batch.get("k_hop_mask"),
        k_hop, L, L, block_size=128, device=node_ids.device,
    )

    # 3) Swap each attention forward and stash the shared block mask.
    for layer in model.model.layers:
        layer.self_attn._flex_block_mask = block_mask
        layer.self_attn.forward = MethodType(_flex_attn_forward, layer.self_attn)
    return block_mask

# .venv/bin/python -m src.models.flex_attn.run_sweep  --kind both  --methods flash flash_nc eager flex  --nodes 128 512 2048  --tpn 2 8 32 128  --k-hops 0 2 4  --orderings rcm  --out-dir src/models/flex_attn/results_h100  --max-tokens=400000


# ── model builders ────────────────────────────────────────────────────────────

def _build_gtlm(spec: GraphSpec, dtype, checkpoint_bias: bool = True):
    from src.models import GTLMLlamaConfig, GTLMLlamaForCausalLM
    from transformers import AutoConfig

    base = AutoConfig.from_pretrained(BASE_MODEL)
    cfg = GTLMLlamaConfig(
        **base.to_dict(),
        spd=True, max_spd=32, magnetic=True, magnetic_dim=32, magnetic_q=0.25,
        k_hop=spec.k_hop, k_hop_directed=spec.k_hop_directed, graph_attn_impl="eager",
        checkpoint_graph_bias=checkpoint_bias,
    )
    model = GTLMLlamaForCausalLM.from_pretrained(
        BASE_MODEL, config=cfg, torch_dtype=dtype, attn_implementation="eager",
    )
    return model.cuda().train()


def _build_flash(dtype):
    # The `flash_attn` pip package isn't installed, so we use PyTorch SDPA, which
    # auto-selects its fused flash kernel for plain causal attention on A100 —
    # the same kernel the isolation bench forces. This is the bias-free floor.
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, attn_implementation="sdpa",
    )
    return model.cuda().train()


# ── one (method, config) run ──────────────────────────────────────────────────

def run_full_model(
    spec: GraphSpec,
    method: str,
    batch_size: int = 1,
    checkpoint_bias: bool = True,
    gradient_checkpointing: bool = False,
    n_warmup: int = 3,
    n_iter: int = 10,
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    result = {
        "kind": "full_model", "method": method, "ok": False, "error": None,
        "spec": dataclasses.asdict(spec),
        "run": {"batch_size": batch_size, "block_size": 128, "base_model": BASE_MODEL,
                "checkpoint_bias": checkpoint_bias,
                "gradient_checkpointing": gradient_checkpointing},
        "shape": {}, "density": {}, "timing_ms": {}, "memory_mb": {},
    }
    try:
        dev = torch.device("cuda")
        batch, meta = make_model_batch(spec, batch_size, dev, dtype=dtype)
        result["shape"] = {k: meta[k] for k in ("seq_len", "max_num_nodes", "total_tokens")}

        # Mask density — computed on the (unpadded) graph for EVERY method so
        # token- and block-level sparsity are comparable across eager/flex/flash.
        d = compute_density(batch["node_ids"], batch["prompt_node"],
                            batch["attention_mask"], batch.get("k_hop_mask"),
                            spec.k_hop, block_size=128)
        result["density"] = d.as_dict()

        # ── inputs per method ──
        compile_ms = None
        if method == "flash":
            model = _build_flash(dtype)
            inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"],
                      "labels": batch["labels"]}
        else:
            model = _build_gtlm(spec, dtype, checkpoint_bias)
        if gradient_checkpointing:
            # Recompute each decoder layer in backward (#9). use_reentrant=False
            # matches (and nests with) the #7 bias checkpoint; use_cache is
            # incompatible with checkpointing and unused in training anyway.
            model.config.use_cache = False
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        if method != "flash":
            if method == "flex":
                batch = _pad_batch_for_flex(batch, block_size=128)
            inputs = {k: v for k, v in batch.items() if k != "labels"}
            inputs["labels"] = batch["labels"]
            if method == "flex":
                t0 = time.perf_counter()
                _install_flex(model, batch, spec.k_hop)
                out0 = model(**inputs)              # triggers flex compile
                out0.loss.backward(); torch.cuda.synchronize()
                model.zero_grad(set_to_none=True)
                compile_ms = (time.perf_counter() - t0) * 1e3
                del out0

        def _fwd():
            with torch.no_grad():
                return model(**inputs).loss

        def _fwd_bwd():
            model.zero_grad(set_to_none=True)
            loss = model(**inputs).loss
            loss.backward()

        # forward-only
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        fwd_ms, fwd_std = _cuda_time(_fwd, n_warmup, n_iter)
        torch.cuda.synchronize(); peak_fwd = _mb(torch.cuda.max_memory_allocated())

        # forward+backward
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        fb_ms, fb_std = _cuda_time(_fwd_bwd, n_warmup, n_iter)
        torch.cuda.synchronize(); peak_fb = _mb(torch.cuda.max_memory_allocated())

        # direct backward timing (preferred over bwd_est: the no-grad forward
        # saves no activations, so fwd_bwd − fwd over-estimates the backward)
        def _bwd_sample() -> float:
            model.zero_grad(set_to_none=True)
            loss = model(**inputs).loss
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(); loss.backward(); end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end)

        for _ in range(n_warmup):
            _bwd_sample()
        bwd_direct = torch.tensor([_bwd_sample() for _ in range(n_iter)])

        result["ok"] = True
        result["timing_ms"] = {
            "fwd": fwd_ms, "fwd_std": fwd_std,
            "fwd_bwd": fb_ms, "fwd_bwd_std": fb_std,
            "bwd": float(bwd_direct.mean()), "bwd_std": float(bwd_direct.std()),
            "bwd_est": fb_ms - fwd_ms,              # legacy estimate; biased
            "compile": compile_ms,                  # flex one-time
            "reorder_mean": meta["reorder_time_mean_s"] * 1e3,
            "reorder_max": meta.get("reorder_time_max_s", 0.0) * 1e3,
        }
        result["memory_mb"] = {"peak_fwd": peak_fwd, "peak_fwd_bwd": peak_fb}
    except torch.cuda.OutOfMemoryError as e:
        result["error"] = "OOM"; result["error_detail"] = str(e)[:200]
    except Exception as e:  # noqa: BLE001
        import traceback
        result["error"] = type(e).__name__
        result["error_detail"] = traceback.format_exc()[-500:]
    finally:
        gc.collect(); torch.cuda.empty_cache()
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["eager", "flex", "flash"])
    p.add_argument("--n-nodes", type=int, required=True)
    p.add_argument("--tokens-per-node", type=int, required=True)
    p.add_argument("--prompt-tokens", type=int, default=128)
    p.add_argument("--k-hop", type=int, default=0)
    p.add_argument("--k-hop-directed", action="store_true")
    p.add_argument("--ordering", default="rcm", choices=["rcm", "random", "identity"])
    p.add_argument("--directed", action="store_true")
    p.add_argument("--magnetic-m", type=int, default=32)
    p.add_argument("--jitter", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--checkpoint-bias", default=True,
                   action=argparse.BooleanOptionalAction,
                   help="gradient-checkpoint the per-layer bias modules (#7); "
                        "--no-checkpoint-bias measures the autograd-saved baseline")
    p.add_argument("--gradient-checkpointing", default=False,
                   action=argparse.BooleanOptionalAction,
                   help="gradient-checkpoint the decoder layers (#9) — trades "
                        "~1.3x latency for several-x activation memory at large L")
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument("--n-iter", type=int, default=10)
    return p.parse_args(argv)


def main(argv=None):
    a = _parse_args(argv)
    spec = GraphSpec(
        n_nodes=a.n_nodes, tokens_per_node=a.tokens_per_node, prompt_tokens=a.prompt_tokens,
        k_hop=a.k_hop, k_hop_directed=a.k_hop_directed, ordering=a.ordering, directed=a.directed,
        magnetic_m=a.magnetic_m, jitter=a.jitter, seed=a.seed,
    )
    res = run_full_model(spec, a.method, batch_size=a.batch_size,
                         checkpoint_bias=a.checkpoint_bias,
                         gradient_checkpointing=a.gradient_checkpointing,
                         n_warmup=a.n_warmup, n_iter=a.n_iter)
    emit_result(res)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
