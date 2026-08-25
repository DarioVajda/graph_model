"""Is the normalized landmark bias actually LIVE, and how large has it grown?

    python -m src.experiments.bias_experiments.landmark.diagnose_norm

042 replaced 040's unbounded bilinear form with the `MIXED_BIAS.md` §5.8 remedy:
L2-normalize each side's per-(node, channel) factor and put the magnitude in a
per-head gain, so Cauchy-Schwarz gives |b| <= n_chan * max|gamma|. That bound is
the whole point of the change, and it is only worth anything if `gamma` actually
LEFT its zero init — a gain stuck at 0 is a bias that is identically 0, which
would train cleanly and read as a clean negative. That is this repo's most
expensive failure mode, so it gets checked directly rather than assumed.

Reported per checkpoint:
  max|gamma|, mean|gamma|  — did the gate open, and how far
  bound = n_chan*max|gamma| — the HARD ceiling on |b|, to compare against the
                              O(1-10) attention logits it is added to
  max|F|, max|G|            — the tables; under the norm their SCALE is a
                              redundant degree of freedom (normalize() divides it
                              out), so these should stay near their exp(-d/tau)
                              init and drift only in SHAPE
  n_layers                  — how many layers reported, as a wiring check

Contrast with 040 (`diagnose_scale.py`): there the same quantity was 9-15 at
bias_lr 5e-3 and 64-240 at 2e-2, i.e. the bias was replacing attention rather
than nudging it. If 042's bound is O(1) and F1 is still below the floor, the
scale hypothesis is spent and the cause lies elsewhere.
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import torch

CKPT_ROOT = "checkpoints/kgqa"


def _latest_bias_file(run_dir: str) -> str | None:
    """The bias tables from the highest-numbered checkpoint under `run_dir`."""
    cands = glob.glob(os.path.join(run_dir, "**", "bias_parameters.pt"), recursive=True)
    if not cands:
        return None

    def step(p: str) -> int:
        m = re.search(r"checkpoint-(\d+)", p)
        return int(m.group(1)) if m else -1

    return max(cands, key=step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="042_webqsp_landmark_norm")
    ap.add_argument("--channels", type=int, default=3)
    args = ap.parse_args()

    dirs = sorted(glob.glob(os.path.join(CKPT_ROOT, f"{args.prefix}*")))
    if not dirs:
        raise SystemExit(f"no checkpoints matching {args.prefix}*")

    print(f"{'run':46s} {'step':>6s} {'max|g|':>8s} {'mean|g|':>8s} "
          f"{'bound':>8s} {'max|F|':>8s} {'max|G|':>8s} {'L':>3s}")
    for d in dirs:
        path = _latest_bias_file(d)
        tag = os.path.basename(d).replace(args.prefix + "_", "")[:46]
        if path is None:
            print(f"{tag:46s} (no bias_parameters.pt yet)")
            continue
        sd = torch.load(path, map_location="cpu")
        gains = [v for k, v in sd.items() if k.endswith(".gain")]
        Fs = [v for k, v in sd.items() if k.endswith(".F")]
        Gs = [v for k, v in sd.items() if k.endswith(".G")]
        if not gains:
            print(f"{tag:46s} NO `gain` KEY -- this checkpoint is the UNNORMALIZED "
                  f"form; landmark_norm did not reach the model")
            continue
        m = re.search(r"checkpoint-(\d+)", path)
        mg = max(float(g.abs().max()) for g in gains)
        ag = sum(float(g.abs().mean()) for g in gains) / len(gains)
        print(f"{tag:46s} {m.group(1) if m else '?':>6s} {mg:8.4f} {ag:8.4f} "
              f"{args.channels * mg:8.3f} "
              f"{max(float(f.abs().max()) for f in Fs):8.4f} "
              f"{max(float(g.abs().max()) for g in Gs):8.4f} {len(gains):3d}")

    print(f"\nbound = {args.channels}*max|gamma| is a HARD ceiling on |b| "
          f"(Cauchy-Schwarz on unit factors).")
    print("Attention logits are q.k/sqrt(64), i.e. O(1-10). A bound of that order "
          "is a nudge;\n040's 64-240 was a replacement. A bound at ~0 means the "
          "gate never opened and the arm is a silent no-op.")


if __name__ == "__main__":
    main()
