"""Recipe replay and checkpoint loading for Phase 0.

Phase 0 measures a *trained* magnetic bias, so it has to reconstruct exactly the
configuration that trained it. Nothing here retypes flags: the sweep's own job
script is pushed back through the experiment's ``build_parser`` /
``config_from_args``, which is the same chain training used (and the chain
``tests/experiments/test_magnetic_groups_cli.py`` pins). A recipe that drifts
therefore fails loudly at parse time instead of silently measuring a different
model than the checkpoint was trained as.

Two things are deliberately *not* taken from the recipe:

* ``magnetic_m`` — Phase 0 sweeps it (see LINEAR_BIAS.md §2.6) by truncating the
  stored eigenvectors at collate time. This is bit-identical to a dataset built
  at that ``m`` because ``eigh`` returns ascending eigenvalues and both the
  builder (``utils/magnetic_lap.py``) and the collator truncate by prefix slice.
* ``pad_to_block`` — flex block alignment pads *tokens*, and Phase 0 only reads
  node-level spectral features. Disabled so no compiled kernel is required.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from typing import Iterator, Optional

import torch

from ....utils import GraphCollatorV2

# Experiment module -> (parser builder, config builder) resolved lazily so that
# importing this module does not drag in every experiment's dependencies.
_EXPERIMENTS = {
    "src.experiments.kgqa": "kgqa",
    "src.experiments.context": "context",
}


def parse_job_script(path: str) -> tuple[str, list[str]]:
    """Return ``(experiment_module, argv)`` from a sweep's generated job script.

    The scripts are generated one-command-per-file by the sweep runner, so the
    single ``python -m <module> ...`` line is the recipe in full.
    """
    with open(path) as fh:
        text = fh.read()
    m = re.search(r"^\s*python\s+-m\s+(\S+)\s+(.*)$", text, re.M)
    if m is None:
        raise ValueError(f"No 'python -m ...' invocation found in {path}")
    module, raw = m.group(1), m.group(2)
    if module not in _EXPERIMENTS:
        raise ValueError(
            f"{path} runs {module!r}, which Phase 0 does not know how to replay. "
            f"Known: {sorted(_EXPERIMENTS)}")
    return module, shlex.split(raw)


def config_from_job(path: str):
    """Rebuild the training ``RunConfig`` from a job script."""
    module, argv = parse_job_script(path)
    if module == "src.experiments.kgqa":
        from ...kgqa.__main__ import build_parser, config_from_args
    else:
        from ...context.__main__ import build_parser, config_from_args
    args = build_parser().parse_args(argv)
    return _EXPERIMENTS[module], config_from_args(args)


# ── Checkpoint weights ────────────────────────────────────────────────────────

_MAG_KEYS = ("lambda_lin.weight", "lambda_lin.bias",
             "deep_set.0.weight", "deep_set.0.bias",
             "proj.0.weight", "proj.0.bias",
             "proj.2.weight", "proj.2.bias")

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.self_attn\.graph_bias\.bias_modules\.(\d+)\.(.+)$")


def resolve_checkpoint(run_dir: str) -> str:
    """Return the newest ``checkpoint-N`` inside a run directory (or the dir itself)."""
    if os.path.isfile(os.path.join(run_dir, "bias_parameters.pt")):
        return run_dir
    cands = [d for d in os.listdir(run_dir) if d.startswith("checkpoint-")]
    if not cands:
        raise FileNotFoundError(f"No checkpoint-* under {run_dir}")
    newest = max(cands, key=lambda d: int(d.split("-")[1]))
    return os.path.join(run_dir, newest)


def load_magnetic_weights(ckpt_dir: str) -> dict[int, dict[str, torch.Tensor]]:
    """Per-layer magnetic-bias parameters, upcast to float64.

    The magnetic module is identified by its *parameter signature* rather than by
    a hard-coded ``bias_modules`` index: the index depends on which other bias
    types are enabled (SPD occupies slot 0 in the g-sweep recipes), and silently
    reading the wrong slot would produce a plausible but meaningless fit.
    """
    path = os.path.join(ckpt_dir, "bias_parameters.pt")
    sd = torch.load(path, map_location="cpu", weights_only=True)

    by_slot: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    for key, val in sd.items():
        m = _LAYER_RE.search(key)
        if m is None:
            continue
        layer, slot, param = int(m.group(1)), int(m.group(2)), m.group(3)
        by_slot.setdefault((layer, slot), {})[param] = val.to(torch.float64)

    out: dict[int, dict[str, torch.Tensor]] = {}
    for (layer, slot), params in sorted(by_slot.items()):
        if set(params) != set(_MAG_KEYS):
            continue                                    # SPD / other bias slots
        if layer in out:
            raise ValueError(
                f"Two magnetic-shaped modules on layer {layer} in {path}; "
                "cannot disambiguate.")
        out[layer] = params
    if not out:
        raise ValueError(
            f"No magnetic bias module found in {path}. Slots present: "
            f"{sorted({(l, s) for l, s in by_slot})}")
    return out


# ── Batches of real graphs ────────────────────────────────────────────────────

@dataclass
class MagneticBatch:
    """The node-level spectral inputs one batch contributes to Phase 0."""
    V_real: torch.Tensor        # (B, N, M)
    V_imag: torch.Tensor        # (B, N, M)
    lambdas: torch.Tensor       # (B, M)
    num_nodes: torch.Tensor     # (B,)

    def truncate(self, m: int) -> "MagneticBatch":
        """Keep the lowest ``m`` eigenpairs — the prefix slice the builder uses."""
        if m <= 0 or m >= self.lambdas.shape[1]:
            return self
        return MagneticBatch(self.V_real[:, :, :m], self.V_imag[:, :, :m],
                             self.lambdas[:, :m], self.num_nodes)

    def to(self, device, dtype=None) -> "MagneticBatch":
        """Move the spectral tensors to a device (``num_nodes`` stays on CPU —
        it is only ever read as Python ints, and keeping it host-side avoids a
        device sync per graph in the inner loop)."""
        cast = (lambda t: t.to(device=device, dtype=dtype)) if dtype is not None \
            else (lambda t: t.to(device=device))
        return MagneticBatch(cast(self.V_real), cast(self.V_imag),
                             cast(self.lambdas), self.num_nodes)


def _dataset_for(kind: str, cfg, split: str):
    if kind == "kgqa":
        from ...kgqa.load_data import load_data
        _, eval_sets, test_sets = load_data(cfg)
        sets = {"dev": eval_sets, "test": test_sets}[split]
        return next(iter(sets.values()))
    # context has no load_data module: splits are loaded individually by name,
    # and its test splits are per-cell (test_n64_t128_h2, ...) rather than one set.
    from ...context.process_dataset import load_split, split_paths
    if split in ("dev", "train"):
        return load_split(cfg, split)
    cells = [n for n in split_paths(cfg) if n.startswith("test")]
    if not cells:
        raise RuntimeError(f"context config defines no test splits; have {split_paths(cfg)}")
    return load_split(cfg, sorted(cells)[0])


def iter_magnetic_batches(
    kind: str, cfg, *, batch_size: int = 2, n_batches: int = 8,
    split: str = "dev", tokenizer=None, seed: int = 0,
) -> Iterator[MagneticBatch]:
    """Yield spectral features from real graphs under the recipe's collator.

    Eigenvectors are emitted at the dataset's stored ``m``; per-``M`` truncation
    is the caller's job (``MagneticBatch.truncate``) so one pass over the data
    serves the whole grid.
    """
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    dataset = _dataset_for(kind, cfg, split)
    collator = GraphCollatorV2(
        tokenizer=tokenizer, k_hop=cfg.k_hop,
        k_hop_directed=getattr(cfg, "k_hop_directed", False),
        magnetic_m=cfg.magnetic_m,          # stored width; truncation happens later
        pad_to_block=False,
        node_position_mode=getattr(cfg, "node_position_mode", "reset"),
        max_spd=cfg.max_spd,
    )

    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(dataset), generator=g).tolist()

    for b in range(n_batches):
        idx = order[b * batch_size:(b + 1) * batch_size]
        if not idx:
            return
        batch = collator([dataset[i] for i in idx])
        V = batch.get("magnetic_V")
        if V is None:
            raise RuntimeError(
                "Collator emitted no magnetic_V — the recipe's magnetic features "
                "are missing, so Phase 0 would measure nothing. Check that the "
                "cached dataset was built with --magnetic-m > 0.")
        yield MagneticBatch(
            V_real=V[..., 0].to(torch.float64),
            V_imag=V[..., 1].to(torch.float64),
            lambdas=batch["magnetic_lambdas"].to(torch.float64),
            num_nodes=batch["num_nodes"],
        )
