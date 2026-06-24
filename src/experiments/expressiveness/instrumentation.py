"""
Instrumentation for the real training runs: speed, memory, and graph sparsity.

Two independent concerns, intentionally kept apart:

  * **Speed + memory** are properties of the *training step*, so they are
    captured live from the real ``Trainer`` loop via :class:`StepMemCallback`
    (per-optimizer-step wall time + peak CUDA memory).

  * **Token / block sparsity** are properties of the *data* (graph structure +
    ``k_hop`` + how the collator packs/pads) and do **not** depend on training,
    so :func:`measure_density` computes them once on a random subset of the
    dataset — never inside the training loop.

Both reuse existing machinery: ``transformers.TrainerCallback`` and
``src.models.flex_attn.density.compute_density`` (no new math here).
"""

import statistics
import time
import random

import torch
from transformers import TrainerCallback

from ...utils import GraphCollatorV2
from ...models.flex_attn.density import compute_density


def _cuda_device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


# ── Live speed + memory from the real training loop ──────────────────────────────

class StepMemCallback(TrainerCallback):
    """Record per-optimizer-step wall time and peak CUDA memory during training.

    ``on_step_begin`` / ``on_step_end`` fire once per *optimizer* step (i.e. after
    gradient accumulation), so each timing is a full effective training step. The
    first step(s) are dropped in :meth:`summary` because they include flex's
    one-time ``torch.compile`` (seconds) rather than steady-state cost.
    """

    def __init__(self):
        self.device = _cuda_device()
        self.step_ms = []
        self._t0 = None

    def on_train_begin(self, args, state, control, **kwargs):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def on_step_begin(self, args, state, control, **kwargs):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._t0 = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self._t0 is None:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.step_ms.append((time.perf_counter() - self._t0) * 1000.0)

    def summary(self, warmup=1):
        """Aggregate steady-state ms/step (dropping ``warmup`` steps) + peak GB."""
        steady = self.step_ms[warmup:] or self.step_ms
        peak_gb = (torch.cuda.max_memory_allocated(self.device) / 1e9
                   if self.device.type == "cuda" else float("nan"))
        return {
            "n_steps": len(self.step_ms),
            "step_ms_mean": statistics.mean(steady) if steady else float("nan"),
            "step_ms_median": statistics.median(steady) if steady else float("nan"),
            "peak_gb": peak_gb,
        }


# ── Standalone sparsity on a random subset (data property, not training) ─────────

def _density_of_batch(out, k_hop, block_size, device):
    """Run ``compute_density`` on a collated batch (moving the needed tensors)."""
    node_ids = out["node_ids"].to(device)
    prompt_node = out["prompt_node"].to(device)
    pad_mask = out["attention_mask"].to(device)
    k_hop_mask = out["k_hop_mask"].to(device) if out.get("k_hop_mask") is not None else None
    return compute_density(node_ids, prompt_node, pad_mask, k_hop_mask, k_hop,
                           block_size=block_size)


def measure_density(dataset, tokenizer, pad_token_id, bias_params, k_hop, k_hop_directed,
                    batch_size, num_sample_graphs=16, num_sample_batches=8,
                    block_size=128, seed=0, device=None):
    """Token + block sparsity for ``k_hop`` on a random subset of ``dataset``.

    * **Token (element) sparsity** — measured per graph with an *unpadded* v2
      collator (``pad_to_block=False``, batch=1 ⇒ ``L`` = real length), so it is a
      backend-independent property of the graph + ``k_hop``.
    * **Block sparsity** — measured on *flex-packed* batches (``pad_to_block=True``)
      of the real ``batch_size``: the fraction of 128×128 blocks flex must still
      compute (its realized skip rate). Reported as the flex-packing value; eager
      is dense and does not skip blocks.
    """
    device = device or _cuda_device()
    magnetic_m = bias_params["magnetic_dim"] if bias_params.get("magnetic") else 0

    element_collator = GraphCollatorV2(
        tokenizer=tokenizer, pad_token_id=pad_token_id, k_hop=k_hop,
        k_hop_directed=k_hop_directed, magnetic_m=magnetic_m, pad_to_block=False)
    block_collator = GraphCollatorV2(
        tokenizer=tokenizer, pad_token_id=pad_token_id, k_hop=k_hop,
        k_hop_directed=k_hop_directed, magnetic_m=magnetic_m, pad_to_block=True)

    rng = random.Random(seed)
    n = len(dataset)

    # Token sparsity: per-graph, unpadded (L = real packed length).
    token_sps, unpadded_Ls = [], []
    for i in rng.sample(range(n), min(num_sample_graphs, n)):
        out = element_collator([dataset[i]])
        d = _density_of_batch(out, k_hop, block_size, device)
        token_sps.append(1.0 - d.element_density)
        unpadded_Ls.append(out["node_ids"].shape[1])

    # Block sparsity: flex-packed batches of the real batch_size (realized skip rate).
    block_sps, padded_Ls = [], []
    bs = min(batch_size, n)
    for _ in range(num_sample_batches):
        idx = rng.sample(range(n), bs)
        out = block_collator([dataset[i] for i in idx])
        d = _density_of_batch(out, k_hop, block_size, device)
        block_sps.append(1.0 - d.block_density)
        padded_Ls.append(out["node_ids"].shape[1])

    def _ms(xs):
        return (statistics.mean(xs) if xs else float("nan"),
                statistics.stdev(xs) if len(xs) > 1 else 0.0)

    tok_mean, tok_std = _ms(token_sps)
    blk_mean, blk_std = _ms(block_sps)
    return {
        "k_hop": k_hop,
        "token_sparsity_mean": tok_mean, "token_sparsity_std": tok_std,
        "block_sparsity_mean": blk_mean, "block_sparsity_std": blk_std,
        "unpadded_L_min": min(unpadded_Ls) if unpadded_Ls else None,
        "unpadded_L_max": max(unpadded_Ls) if unpadded_Ls else None,
        "padded_L_max": max(padded_Ls) if padded_Ls else None,
        "n_graphs": len(token_sps), "n_batches": len(block_sps),
    }


# ── Reporting ────────────────────────────────────────────────────────────────────

def format_results_table(rows):
    """Render a combined metrics table.

    Each row is a dict of pre-formatted strings with keys:
    ``config, accuracy, ms_step, peak_gb, token_sp, block_sp``.
    """
    cols = [
        ("config", "config", 20),
        ("accuracy", "accuracy", 16),
        ("ms_step", "ms/step", 14),
        ("peak_gb", "peak GB", 10),
        ("token_sp", "token sp", 14),
        ("block_sp", "block sp", 14),
    ]
    header = " | ".join(f"{h:<{w}}" for _, h, w in cols)
    lines = ["=" * len(header), header, "-" * len(header)]
    for r in rows:
        lines.append(" | ".join(f"{str(r.get(k, '')):<{w}}" for k, _, w in cols))
    lines.append("=" * len(header))
    return "\n".join(lines)
