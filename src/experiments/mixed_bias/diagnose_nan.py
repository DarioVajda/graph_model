"""Per-parameter-group gradient diagnostic for the 020 hybrid divergence.

    python -m src.experiments.mixed_bias.diagnose_nan <exact flags of the failing run>

Arm 4 (`magnetic_hybrid`) at bias_lr 2e-2 diverged on the 4k context task: grad
norms healthy (~10) to epoch 0.13, then 7 282 / 10 596 / 73 371 at epoch 0.14-0.18,
a recovery to ~11, further spikes, and `nan` at epoch 0.27. Three explanations
survive the evidence so far (README, "Divergences") and they make DIFFERENT
predictions about WHICH parameters blow up first:

  * shared-trunk coupling  -> `trunk` (lambda_lin, deep_set) goes first; arm 4 is
                              the only arm whose trunk feeds two heads.
  * the magnitude channel  -> `magnitude_*` goes first, and the trunk only follows
                              once the bad signal has propagated back into it.
  * a pathological sample  -> EVERYTHING spikes on the same single step, with no
                              group leading, and the step is reproducible by index.

So this logs, for every optimizer step, the gradient norm of each group SEPARATELY,
plus a per-group count of non-finite entries.

Two properties this file is built around:

1. It changes NOTHING about the training math. It monkeypatches
   `torch.nn.utils.clip_grad_norm_` to read gradients on the way past — that is the
   one point where every gradient exists, is fully accumulated, and is NOT yet
   scaled — and then calls the original. Reading `.grad` in a Trainer callback
   would be too late: HF clips before `on_pre_optimizer_step`, so every group
   would report a post-clip norm of ~1 and the diagnostic would say nothing.

2. It does NOT shorten the run via `--max-steps`. `max_steps` feeds the LR
   scheduler's total and `warmup_steps = total_steps // 10`, so truncating the run
   would change the LR at every step and the thing under investigation (a
   divergence that began mid-warmup at an instantaneous bias_lr of 0.0045) would
   not reproduce. Instead a callback flips `control.should_training_stop` once
   DIAG_MAX_STEPS optimizer steps have run, which leaves the schedule untouched.

Env:
    DIAG_OUT        jsonl output path (default: results/diagnostic/grad_norms.jsonl)
    DIAG_MAX_STEPS  stop after this many optimizer steps (default 700; the NaN
                    landed near step 550, so this clears it with margin)
"""

from __future__ import annotations

import json
import os
import sys

import torch
import transformers

OUT = os.environ.get(
    "DIAG_OUT",
    "src/experiments/mixed_bias/results/diagnostic/grad_norms.jsonl")
MAX_STEPS = int(os.environ.get("DIAG_MAX_STEPS", "700"))

# Substring -> group. Ordered: the first match wins, so `magnitude_mlp` is
# claimed before a broader pattern could take it.
_GROUP_PATTERNS = [
    ("magnitude_q_scale", "magnitude_q_scale"),   # the zero-init'd query side
    ("magnitude_k_mix", "magnitude_k_mix"),       # the per-KV-group W_K
    ("magnitude_mlp", "magnitude_mlp"),           # MLP_magnitude
    ("proj", "phase_head"),                       # LinearMagneticBias's head
    ("lambda_lin", "trunk"),                      # \
    ("deep_set", "trunk"),                        # / the SHARED DeepSets trunk
    ("lora_", "lora"),                            # the adapter, as a control
]

_state: dict = {"trainer": None, "step": 0, "groups": None, "fired": 0}


def _classify(name: str) -> str:
    for pat, group in _GROUP_PATTERNS:
        if pat in name:
            return group
    return "other"


def _build_groups(model):
    groups: dict[str, list] = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            groups.setdefault(_classify(name), []).append((name, p))
    return groups


def _log_group_norms():
    trainer = _state["trainer"]
    if trainer is None:
        return
    model = trainer.model
    if _state["groups"] is None:
        _state["groups"] = _build_groups(model)
        summary = {g: len(v) for g, v in _state["groups"].items()}
        print(f"[diag] parameter groups: {summary}", flush=True)

    rec = {"step": _state["step"]}
    for gname, params in _state["groups"].items():
        sq, nonfinite, worst, worst_name = 0.0, 0, 0.0, None
        for name, p in params:
            if p.grad is None:
                continue
            g = p.grad.detach().float()
            finite = bool(torch.isfinite(g).all())
            if not finite:
                nonfinite += 1
            v = float(g.norm())
            if v == v and v > worst:      # NaN-safe max
                worst, worst_name = v, name
            if v == v:
                sq += v * v
        rec[gname] = {"norm": round(sq ** 0.5, 4), "nonfinite_tensors": nonfinite,
                      "worst": round(worst, 4), "worst_param": worst_name}
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    _state["step"] += 1
    _state["fired"] += 1


_orig_clip = torch.nn.utils.clip_grad_norm_


def _patched_clip(parameters, max_norm, *args, **kwargs):
    try:
        _log_group_norms()
    except Exception as e:                       # never let the probe kill the run
        print(f"[diag] logging failed at step {_state['step']}: {e!r}", flush=True)
    return _orig_clip(parameters, max_norm, *args, **kwargs)


torch.nn.utils.clip_grad_norm_ = _patched_clip
# accelerate resolves `torch.nn.utils.clip_grad_norm_` at call time, so the line
# above is enough for the normal path; rebind the module attribute too in case a
# version imported the symbol directly at import time.
try:
    import accelerate.accelerator as _acc
    if getattr(_acc, "clip_grad_norm_", None) is not None:
        _acc.clip_grad_norm_ = _patched_clip
except Exception:
    pass


class _StopAfter(transformers.TrainerCallback):
    """Stop after MAX_STEPS without touching the LR schedule (see module docstring)."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= MAX_STEPS:
            print(f"[diag] reached DIAG_MAX_STEPS={MAX_STEPS}; stopping.", flush=True)
            control.should_training_stop = True
        return control


_orig_init = transformers.Trainer.__init__


def _patched_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    _state["trainer"] = self
    self.add_callback(_StopAfter())


transformers.Trainer.__init__ = _patched_init


def main():
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    print(f"[diag] writing per-group grad norms to {OUT}", flush=True)
    print(f"[diag] will stop after {MAX_STEPS} optimizer steps", flush=True)

    from src.experiments.context.__main__ import main as context_main
    try:
        context_main()
    finally:
        # A probe that silently never fired would produce an empty file and look
        # like "no anomaly found", which is the one conclusion it must not support.
        if _state["fired"] == 0:
            print("[diag] *** THE CLIP HOOK NEVER FIRED — no data was collected. ***",
                  file=sys.stderr, flush=True)
            raise SystemExit(3)
        print(f"[diag] logged {_state['fired']} optimizer steps to {OUT}", flush=True)


if __name__ == "__main__":
    main()
