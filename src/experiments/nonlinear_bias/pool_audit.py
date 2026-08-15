"""Audit whether `magnetic_nonlinear`'s head actually engaged, before any null is read.

`mixed_bias`'s arm v2 established the discipline: its gate was bit-identical to
arm 2 at initialisation, so its null was only interpretable once `gate_audit.py`
proved the gate had left identity. The analogue here is NOT the same quantity,
and the difference matters:

  * **gamma_in is the gate.** It is zero-initialised, and at zero the bias is
    IDENTICALLY ZERO. A run whose gamma_in never left 0 trained with no graph
    bias at all and its "null" is worth nothing. This is the check that must pass
    before anything else is read, and it is parameter-only.
  * **the pool is NOT inert at init.** W_attn is randomly initialised, so the
    pool is already non-uniform at step 0 — unlike v2's gate, the mechanism
    engages by construction and the question is whether it LEARNED, i.e. whether
    W_attn moved off its initialisation scale.
  * **the ablation is the real control.** `magnetic_pool: "uniform"` is in the
    grid precisely so "did the learned pool buy anything" is answered
    experimentally rather than by a proxy statistic. This script is the cheap
    check that runs first; arm N vs arm N-uniform is the answer.

Part B additionally reports the REALISED pool entropy on the run's own dataset,
which is the direct measurement: H(w)/log n_b == 1.0 is a pool that never left
uniform. It loads only the trunk and one pooling head — no LLM weights — so it is
seconds of CPU, not a GPU job.

    python3 -m src.experiments.nonlinear_bias.pool_audit <checkpoint-glob> [--data DIR]
"""

from __future__ import annotations

import argparse
import glob
import math
import statistics as st
import sys

import torch


def _shapes(sd):
    """Recover (magnetic_dim, d_struct, H_Q, H_KV, head_dim) from the checkpoint.

    Read off the tensors rather than taken from a config, so the audit cannot
    disagree with the weights it is auditing.
    """
    w_val_out = next(v for k, v in sd.items() if k.endswith("W_val_out"))
    w_val_in = next(v for k, v in sd.items() if k.endswith("W_val_in"))
    lam = next(v for k, v in sd.items() if k.endswith("lambda_lin.weight"))
    h_q, m, d = w_val_out.shape
    return m, d, h_q, w_val_in.shape[0], lam.shape[0]


def _per_layer(sd, suffix):
    return [v for k, v in sd.items() if k.endswith(suffix)]


def audit_params(name, ck, sd):
    """Part A — parameter-only. Runs anywhere, needs no data."""
    m, d, h_q, h_kv, head_dim = _shapes(sd)
    g_out = _per_layer(sd, "gamma_out")
    g_in = _per_layer(sd, "gamma_in")
    a_out = _per_layer(sd, "W_attn_out")

    print(f"{name}  [{ck.rsplit('/', 1)[1]}, {len(g_in)} layers, "
          f"m={m} d_struct={d} H_Q={h_q} H_KV={h_kv}]")

    # THE gate check. gamma_in == 0 => the bias was identically zero all run.
    inf_in = [float(t.float().abs().max()) for t in g_in]
    print(f"   |gamma_in|_inf   min {min(inf_in):9.5f}  med {st.median(inf_in):9.5f}  "
          f"max {max(inf_in):9.5f}   (exactly 0.0 at init)")
    if max(inf_in) == 0.0:
        print("   *** gamma_in NEVER LEFT ZERO — this run had NO graph bias. "
              "Its result is not a null, it is an untrained module. ***")

    inf_out = [float(t.float().abs().max()) for t in g_out]
    print(f"   |gamma_out|_inf  min {min(inf_out):9.5f}  med {st.median(inf_out):9.5f}  "
          f"max {max(inf_out):9.5f}   (exactly 1.0 at init)")

    # The §2.4 range bound on |b|, per layer: max|g_out| * max|g_in| * d_struct.
    bound = [a * b * d for a, b in zip(inf_out, inf_in)]
    print(f"   gain bound       min {min(bound):9.4f}  med {st.median(bound):9.4f}  "
          f"max {max(bound):9.4f}   (|b| <= this)")

    if a_out:
        # W_attn ~ U(-1/sqrt(m), 1/sqrt(m)) at init, so E||W||_F = sqrt(H/3).
        init_f = math.sqrt(h_q / 3.0)
        n = [float(t.float().norm()) for t in a_out]
        print(f"   ||W_attn_out||_F min {min(n):9.4f}  med {st.median(n):9.4f}  "
              f"max {max(n):9.4f}   (~{init_f:.4f} at init)")
        if abs(st.median(n) - init_f) < 0.02 * init_f:
            print("   NOTE: W_attn is within 2% of its initialisation norm. The pool is "
                  "still non-uniform (it was at init), but it may not have LEARNED — read "
                  "the uniform-pool arm, which is the control for exactly this.")
    else:
        print("   W_attn: absent — this is the uniform-pool ABLATION arm, as intended.")


def audit_entropy(name, sd, data_dir, n_items=32, magnetic_m=64):
    """Part B — realised pool entropy on the run's own graphs.

    Loads the trunk and ONE pooling head out of the checkpoint and runs them on
    real batches. No LLM weights are touched, so this is seconds of CPU.
    """
    from src.models.bias import MagneticNonlinearBias, MagneticPairTrunk
    from src.utils.text_graph_dataset import TextGraphDataset
    from src.utils.text_graph_collator_v2 import GraphCollatorV2

    m, d, h_q, h_kv, head_dim = _shapes(sd)

    class _Cfg:
        magnetic_dim, magnetic_struct_dim = m, d
        num_key_value_heads, bias_self_node = h_kv, True
        magnetic_pool = "attn" if any(k.endswith("W_attn_out") for k in sd) else "uniform"

    trunk = MagneticPairTrunk(h_q, head_dim, _Cfg()).double().eval()
    trunk.load_state_dict({k.split("shared_graph_bias_trunk.")[1]: v.double()
                           for k, v in sd.items() if "shared_graph_bias_trunk" in k})

    # Layer 0's head. Every layer pools the same E; one is enough to say whether
    # the pool moved, and reporting all 16 would bury the number.
    pre = next(k.rsplit(".", 1)[0] for k in sd if k.endswith("W_val_out"))
    head = MagneticNonlinearBias(h_q, head_dim, _Cfg()).double().eval()
    head.load_state_dict({k.split(pre + ".")[1]: v.double()
                          for k, v in sd.items() if k.startswith(pre + ".")})
    head._record_pool_stats = True

    ds = TextGraphDataset.load(data_dir)
    items = [ds[i] for i in range(min(n_items, len(ds)))]
    batch = GraphCollatorV2(pad_token_id=0, magnetic_m=magnetic_m)(items)

    E = trunk(dtype=torch.float64, device=torch.device("cpu"),
              magnetic=(batch["magnetic_V"].double(), batch["magnetic_lambdas"].double()),
              num_nodes=batch["num_nodes"])
    head(dtype=torch.float64, device=torch.device("cpu"),
         num_nodes=batch["num_nodes"], pair_features=E)

    ent = head._pool_stats
    print(f"   pool entropy     out {ent['out']:.5f}   in {ent['in']:.5f}   of log(n_b)   "
          f"(1.0 = never left uniform; {len(items)} real graphs)")
    if min(ent.values()) > 0.999:
        print("   *** the pool is indistinguishable from uniform — a null from this run "
              "says nothing about the learned pool, only about row summaries. ***")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("patterns", nargs="*",
                   default=["checkpoints/kgqa/032_webqsp_nonlinear_*"])
    p.add_argument("--data", default=None,
                   help="a built dataset split (…/dev.gtds) for the entropy report")
    p.add_argument("--magnetic-m", type=int, default=64)
    a = p.parse_args(argv)

    seen = 0
    for pat in a.patterns:
        for run_dir in sorted(glob.glob(pat)):
            cks = sorted(glob.glob(run_dir + "/checkpoint-*"),
                         key=lambda q: int(q.rsplit("-", 1)[1]))
            if not cks:
                print(f"{run_dir}: no checkpoints")
                continue
            ck = cks[-1]
            sd = torch.load(ck + "/bias_parameters.pt", map_location="cpu",
                            weights_only=True)
            if not any(k.endswith("gamma_in") for k in sd):
                print(f"{run_dir.split('/')[-1]}: NO gamma_in — not this arm. "
                      f"keys: {list(sd)[:6]}")
                continue
            audit_params(run_dir.split("/")[-1], ck, sd)
            if a.data:
                try:
                    audit_entropy(run_dir.split("/")[-1], sd, a.data,
                                  magnetic_m=a.magnetic_m)
                except Exception as e:               # never let part B hide part A
                    print(f"   entropy report unavailable: {type(e).__name__}: {e}")
            print()
            seen += 1
    if not seen:
        print("no runs matched", a.patterns)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
