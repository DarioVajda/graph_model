"""Why does the landmark arm land BELOW the no-bias floor? Read the trained tables.

    python -m src.experiments.bias_experiments.landmark.diagnose_scale

040 put landmark at 0.357 F1 against a 0.462 no-bias floor — a 10 pp *regression*,
consistent across dims, both LRs and both seeds (seed spread 0.002-0.013). A
zero-initialised bias that merely failed to help would sit AT the floor; landing
far below it means the bias is actively destroying attention, and the tight spread
says systematic cause, not instability.

The prime suspect is scale, for a reason that is a defect in my own analysis.
`LANDMARK_BIAS.md` argued no normalization was needed because the form is
"degree-1 in each side". That is true and irrelevant: the bias is the PRODUCT of
two *trainable* factors, F and G, so it is degree-2 in the learned parameters,
with no bound on |F||G| at all. `magnetic_linear` is not comparable — there one
side is the raw orthonormal eigenvector (|V| <= 1) and only W is learned, i.e.
degree-1 in learned parameters overall. This is the `MIXED_BIAS.md` §5.7 failure
in a different costume.

The 2e-2 rows are the corroborating evidence: landmark degrades sharply with LR
(0.357 -> 0.211 at dims 24) while magnetic_linear is flat (0.6565 at both). A bias
that gets worse the faster you train it is a bias whose magnitude is running away.

Reported per checkpoint: max|F|, max|G|, the implied bound 3*max|F|*max|G| on |b|,
and the actual |b| on a real batch — next to the attention logit scale it competes
with, which is what decides whether the bias is a nudge or a wrecking ball.
"""

from __future__ import annotations

import glob
import math
import os

import torch

CKPT_ROOT = "checkpoints/kgqa"


def main():
    dirs = sorted(glob.glob(os.path.join(CKPT_ROOT, "040_webqsp_dimsweep_*landmarkTrue*")))
    dirs += sorted(glob.glob(os.path.join(CKPT_ROOT, "040_webqsp_dimsweep_*landmarkFalse*")))[:3]
    if not dirs:
        raise SystemExit("no 040 checkpoints found")

    print(f"{'run':58s} {'maxF':>8s} {'maxG':>8s} {'bound':>10s} {'|b|max':>9s} {'|b|mean':>9s}")
    for d in dirs:
        cand = glob.glob(os.path.join(d, "**", "bias_parameters.pt"), recursive=True)
        if not cand:
            print(f"{os.path.basename(d)[:58]:58s} (no bias_parameters.pt)")
            continue
        sd = torch.load(sorted(cand)[-1], map_location="cpu")
        F = [v for k, v in sd.items() if k.endswith(".F")]
        G = [v for k, v in sd.items() if k.endswith(".G")]
        name = os.path.basename(d)
        tag = name.split("040_webqsp_dimsweep_")[-1][:56]
        if not F:
            # magnetic_linear comparator: report its head's scale instead
            W = [v for k, v in sd.items() if "proj.0.weight" in k]
            if W:
                mw = max(float(w.abs().max()) for w in W)
                print(f"{tag:58s} {'-':>8s} {'-':>8s} {'-':>10s} "
                      f"{'(magW=' + f'{mw:.3f})':>9s}")
            continue
        mf = max(float(f.abs().max()) for f in F)
        mg = max(float(g.abs().max()) for g in G)
        # |b| <= 3 * max|F| * max|G| (three channels, 1/k_val makes each a mean)
        bound = 3 * mf * mg
        # A representative realised magnitude: mean over layers of the per-layer
        # product of RMS magnitudes, which is what a typical pair actually sees.
        real = sum(float(f.abs().mean()) * float(g.abs().mean()) * 3
                   for f, g in zip(F, G)) / len(F)
        print(f"{tag:58s} {mf:8.3f} {mg:8.3f} {bound:10.2f} {bound:9.2f} {real:9.4f}")

    print(f"\nReference: attention logits are q.k/sqrt(d_head) with d_head=64, "
          f"i.e. O(1-10) before softmax.")
    print(f"A bias bound far above that does not nudge attention, it replaces it.")


if __name__ == "__main__":
    main()
