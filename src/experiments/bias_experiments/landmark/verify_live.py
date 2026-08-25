"""Prove the landmark bias is LIVE on real data, end to end, before the sweep.

    python -m src.experiments.bias_experiments.landmark.verify_live          # WebQSP
    python -m src.experiments.bias_experiments.landmark.verify_live --dataset graphqa

The correctness gate proves the math on synthetic tensors; the smoke sweep proves
the pipeline does not crash. Neither proves the bias is non-zero on a real batch —
and a wired-but-inert bias is this repo's most expensive failure mode: it trains
cleanly, scores like the floor, and reads as "landmark does not work".
`feedback-verify-nulls-are-real` says prove a zero-init module left its init
before reporting a null; this does it *before* spending the GPU-hours, not after.

It has already earned its keep once: the in-place `clamp_` that mutated the
caller's feature tensor was invisible to both the synthetic gate and a 30-step
smoke run, and showed up only here.

Asserted, on a real batch pulled through the real collator:
  1. the `landmark` feature actually reaches the model (not None, right shape);
  2. the bias is exactly 0 at init (as designed) — so a non-zero later is
     attributable to training and not to a mis-initialisation;
  3. the TWO-STEP UNROLL the spec claims really happens: at init only `gain` has
     a gradient (with gain = 0 the query factor is 0, so the tables cannot move
     yet); after one step on `gain` the bias is non-zero AND the tables start
     receiving gradient. Checking `G` first, as this script used to, asserts
     something that is false by construction under the normalized form;
  4. the dense forward equals the factorized inner product on that real batch;
  5. slicing k changes the bias — i.e. `landmark_k_collate` is not being ignored,
     which would silently collapse the whole dimension sweep to one point.

GraphQA is worth running separately from WebQSP even though the module is shared:
its graphs average 12.9 nodes against a stored k of 16, so most rows are heavily
PAD-padded. PAD inertness is unit-tested, but this is the only place it is
exercised at the padding density the real GraphQA data actually has.
"""

from __future__ import annotations

import argparse

import torch

from ....utils.text_graph_collator_v2 import GraphCollatorV2
from ....utils.text_graph_dataset import TextGraphDataset


DATASETS = {
    "webqsp": dict(
        cache=("src/experiments/kgqa/processed_datasets/"
               "sr-webqsp_meta-llama-Llama-3.2-1B_vlast_1_cap512_nmax50_ver8"
               "_spd64_magq0.25m128_len1024_rcm1_seed42_dfv3_qnisolated/test"),
        k=32, k_small=8, d_max=8, model="meta-llama/Llama-3.2-1B"),
    "graphqa": dict(
        cache=("src/experiments/graphqa/processed_datasets/standard/"
               "shortest_path__q0.25_rw16_len1024_qn-isolated/test"),
        k=16, k_small=4, d_max=8, model="meta-llama/Llama-3.2-1B"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="webqsp")
    args = ap.parse_args()
    spec = DATASETS[args.dataset]

    from transformers import AutoTokenizer
    from ....models.bias import LandmarkBias

    tok = AutoTokenizer.from_pretrained(spec["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = TextGraphDataset.load(spec["cache"])
    coll = GraphCollatorV2(tokenizer=tok, k_hop=0, magnetic_m=0, pad_to_block=False,
                           landmark_d_max=spec["d_max"], landmark_required=True)
    batch = coll([ds[i] for i in range(4)])

    lm = batch.get("landmark")
    assert lm is not None, "(1) FAILED: collator emitted no landmark tensor"
    B, N, C, k = lm.shape
    assert C == 3 and k == spec["k"], f"(1) FAILED: shape {tuple(lm.shape)}"
    pad_frac = (lm == spec["d_max"] + 2).double().mean().item()
    print(f"(1) OK  landmark reaches the model: {tuple(lm.shape)}, "
          f"dtype={lm.dtype}, symbols {int(lm.min())}..{int(lm.max())}, "
          f"PAD frac {pad_frac:.3f}")

    class Cfg:
        landmark, landmark_k = True, spec["k"]
        landmark_k_collate = 0
        landmark_d_max, landmark_tau, landmark_channels = spec["d_max"], 2.0, 3
        landmark_norm, landmark_gain_scale = True, 1.0
        bias_self_node = True

    mod = LandmarkBias(32, 64, Cfg()).double()
    dev = torch.device("cpu")
    b0 = mod(dtype=torch.float64, device=dev, landmark=lm)
    assert b0.abs().max().item() == 0.0, f"(2) FAILED: init bias {b0.abs().max()}"
    print("(2) OK  bias is exactly 0 at init on a real batch")

    # (3) the two-step unroll, on real data.
    def _loss(m):
        b = m(dtype=torch.float64, device=dev, landmark=lm)
        return b.pow(2).mean() - b.mean()

    _loss(mod).backward()
    gain_g = mod.gain.grad.abs().max().item()
    tbl_g = max(mod.F.grad.abs().max().item(), mod.G.grad.abs().max().item())
    assert gain_g > 0, "(3) FAILED: gain received no gradient — the arm is DEAD"
    assert tbl_g == 0, (f"(3) FAILED: tables moved at step 0 (|g|={tbl_g:.2e}); "
                        "with gain = 0 the query factor is 0, so this means the "
                        "gain is not actually gating the bias")
    torch.optim.SGD(mod.parameters(), lr=1e-2).step()
    b1 = mod(dtype=torch.float64, device=dev, landmark=lm)
    assert b1.abs().max().item() > 0, "(3) FAILED: bias still 0 after a step"
    mod.zero_grad()
    _loss(mod).backward()
    tbl_g2 = max(mod.F.grad.abs().max().item(), mod.G.grad.abs().max().item())
    assert tbl_g2 > 0, "(3) FAILED: tables still frozen after the gain opened"
    print(f"(3) OK  step0 |dgain|={gain_g:.3e}, tables 0 -> step1 |b|max="
          f"{b1.abs().max().item():.3e}, |dtable|={tbl_g2:.3e}")

    # (4) the factorization holds on this real batch, not just on random ints
    q, kk = mod.structural_factors(lm, dtype=torch.float64)
    ref = torch.einsum('bhnc,bmc->bhnm', q, kk)
    err = (b1 - ref).abs().max().item()
    assert err < 1e-10, f"(4) FAILED: dense vs factorized differ by {err}"
    print(f"(4) OK  dense == factorized on real data (max err {err:.2e})")

    # (5) k_collate must change the bias, or the dimension sweep is one point
    Cfg.landmark_k_collate = spec["k_small"]
    mods = LandmarkBias(32, 64, Cfg()).double()
    with torch.no_grad():
        mods.F.copy_(mod.F); mods.G.copy_(mod.G); mods.gain.copy_(mod.gain)
    bs = mods(dtype=torch.float64, device=dev, landmark=lm)
    d = (bs - b1).abs().max().item()
    assert d > 1e-8, (f"(5) FAILED: k={spec['k_small']} and k={spec['k']} give the "
                      "SAME bias — landmark_k_collate is being ignored and the "
                      "whole dimension sweep would collapse to a single point.")
    print(f"(5) OK  k={spec['k_small']} differs from k={spec['k']} "
          f"(max |diff| {d:.3e})")

    print(f"\nALL LIVE CHECKS PASSED on {args.dataset} — the bias is real on real data.")


if __name__ == "__main__":
    main()
