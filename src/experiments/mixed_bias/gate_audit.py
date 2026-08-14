"""Audit how far `magnetic_linear_v2`'s gate moved away from identity.

The arm is bit-identical to `magnetic_linear` when `g == 1`, which is exactly the
state it is initialised in (zero-init output layer). A null result is therefore
only interpretable if the gate actually left 1 — otherwise `025` measured arm 2
under a different name. This reads the saved `bias_parameters.pt` and reports,
per layer, the pre-activation reach of the gate MLP.

`g = 1 + tanh(W2 @ silu(W1 S + b1) + b2)`, so the deviation from identity is
bounded by `|tanh(.)| <= 1` and is zero iff the pre-activation is zero. We report
`||W2||_F` and `|b2|_inf` (both exactly 0 at init) plus, when a features file is
supplied, the realised gate distribution on actual self-energies.

    python3 -m src.experiments.mixed_bias.gate_audit <checkpoint-glob>
"""

import glob
import re
import statistics as st
import sys

import torch


def audit(run_dir: str) -> None:
    cks = sorted(glob.glob(run_dir + "/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
    if not cks:
        print(f"{run_dir}: no checkpoints")
        return
    ck = cks[-1]
    sd = torch.load(ck + "/bias_parameters.pt", map_location="cpu", weights_only=True)
    gate = {k: v for k, v in sd.items() if "gate_mlp" in k}
    if not gate:
        print(f"{run_dir.split('/')[-1]}: NO gate_mlp tensors. keys: {list(sd)[:8]}")
        return

    w2 = [v for k, v in gate.items() if k.endswith("gate_mlp.2.weight")]
    b2 = [v for k, v in gate.items() if k.endswith("gate_mlp.2.bias")]
    w0 = [v for k, v in gate.items() if k.endswith("gate_mlp.0.weight")]

    n2 = [float(t.float().norm()) for t in w2]
    nb = [float(t.float().abs().max()) for t in b2]
    n0 = [float(t.float().norm()) for t in w0]

    # Worst-case pre-activation reach for a unit-norm hidden vector: ||W2||_2 + |b2|_inf.
    reach = [float(torch.linalg.matrix_norm(t.float(), 2)) for t in w2]

    name = run_dir.split("/")[-1]
    print(f"{name}  [{ck.rsplit('/', 1)[1]}, {len(w2)} layers, W2 {tuple(w2[0].shape)}]")
    print(
        f"   ||W2||_F   min {min(n2):8.4f}  med {st.median(n2):8.4f}  max {max(n2):8.4f}"
        f"   (exactly 0.0 at init)"
    )
    print(f"   ||W2||_2   min {min(reach):8.4f}  med {st.median(reach):8.4f}  max {max(reach):8.4f}")
    print(f"   |b2|_inf   min {min(nb):8.4f}  med {st.median(nb):8.4f}  max {max(nb):8.4f}")
    print(f"   ||W1||_F   min {min(n0):8.4f}  med {st.median(n0):8.4f}  max {max(n0):8.4f}")


if __name__ == "__main__":
    pats = sys.argv[1:] or ["checkpoints/kgqa/025_webqsp_linear_v2_*"]
    for pat in pats:
        for d in sorted(glob.glob(pat)):
            audit(d)
