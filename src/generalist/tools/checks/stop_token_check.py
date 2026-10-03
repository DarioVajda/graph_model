"""Is this backbone's stop token one the model can actually be taught to emit?

`adapters/molecules.py::GENERATIVE_ANSWER_KINDS` supervises ``tokenizer.eos_token_id``
at the end of a `text` or `smiles` answer. That is the right token to follow —
the tokenizer knows what its own checkpoint ends with — but following it blindly
across a backbone change is how a run ends up training toward a token the frozen
head cannot produce and the generation loop does not stop on. This checks the
three things that have to line up, and it is a minute of CPU:

1. **Which token is it.** A *base* Llama ends a document with ``<|end_of_text|>``;
   an *Instruct* checkpoint ends a turn with ``<|eot_id|>``. Both ids exist in
   both vocabularies, so the vocabulary does not tell you which model you have —
   ``eos_token_id`` does.
2. **Does `generate` stop on it.** ``generation_config.eos_token_id`` is what
   ends a generation, and nothing checks that it agrees with the token training
   supervised. If they disagree the model learns to stop and then runs on anyway.
3. **Is its row trained.** Llama ties ``lm_head`` to ``embed_tokens``, and the
   LoRA recipe here targets the attention and MLP projections only — so that
   matrix is FROZEN. A token whose row was never trained cannot be emitted no
   matter how much the adapters want it: the reserved special tokens of a base
   checkpoint are exactly that, and picking one as a stop token would reproduce
   the defect it is meant to fix, silently.

Usage (CPU, through Slurm — the safetensors read is small but not login-node work):

    RUNMOD=src.generalist.tools.checks.stop_token_check src/generalist/tools/launch/run_cli.sh
    RUNMOD=src.generalist.tools.checks.stop_token_check src/generalist/tools/launch/run_cli.sh \
        --model-name meta-llama/Llama-3.1-8B
"""

from __future__ import annotations

import argparse
import glob
import json
import os

#: Ids that are the same in every Llama-3 vocabulary, base or Instruct. Named so
#: the report can say which one a checkpoint chose rather than printing a number.
LLAMA3_SPECIALS = {
    128000: "<|begin_of_text|>",
    128001: "<|end_of_text|>",
    128002: "<|reserved_special_token_0|>",
    128004: "<|finetune_right_pad_id|>",
    128008: "<|eom_id|>",
    128009: "<|eot_id|>",
    128010: "<|python_tag|>",
}


def _args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-name", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--n-real", type=int, default=128000,
                   help="ids below this are ordinary tokens, used as the baseline.")
    return p.parse_args(argv)


def _snapshot(model_name: str) -> str:
    """The local HF snapshot directory, or "" when the model is not cached."""
    stem = "models--" + model_name.replace("/", "--")
    root = os.path.join(
        os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"),
        "hub", stem, "snapshots")
    found = sorted(glob.glob(os.path.join(root, "*")))
    return found[-1] if found else ""


def embedding_norms(snapshot: str):
    """Row L2 norms of the tied input/output embedding, straight off disk."""
    import torch
    from safetensors import safe_open

    shards = sorted(glob.glob(os.path.join(snapshot, "*.safetensors")))
    if not shards:
        raise SystemExit(f"no safetensors under {snapshot}")
    for shard in shards:
        with safe_open(shard, framework="pt") as fh:
            for key in fh.keys():
                if key.endswith("embed_tokens.weight") or key.endswith("lm_head.weight"):
                    return key, torch.linalg.vector_norm(
                        fh.get_tensor(key).float(), dim=1)
    raise SystemExit("no embedding tensor found in the checkpoint shards")


def main(argv=None) -> int:
    from transformers import AutoTokenizer

    args = _args(argv)
    snapshot = _snapshot(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    eos = tokenizer.eos_token_id
    print(f"model {args.model_name}")
    print(f"  tokenizer.eos_token_id  {eos}  {tokenizer.eos_token!r}"
          f"   <- what the build supervises")

    if not snapshot:
        print("  not cached locally; the row check needs the weights")
        return 1

    generation = os.path.join(snapshot, "generation_config.json")
    stops = None
    if os.path.exists(generation):
        with open(generation) as fh:
            stops = json.load(fh).get("eos_token_id")
        stops = stops if isinstance(stops, list) else [stops]
        print(f"  generation_config       {stops}   <- what `generate` stops on")
        if eos not in stops:
            print("  MISMATCH: training would supervise a token generation does "
                  "not stop on, so a correct answer still runs to the cap")
            return 1

    key, norms = embedding_norms(snapshot)
    real = norms[: args.n_real]
    mean = real.mean().item()
    print(f"\n  {key}, {norms.shape[0]} rows")
    print(f"  ordinary tokens (0..{args.n_real - 1}): mean norm {mean:.4f}, "
          f"median {real.median().item():.4f}")
    print(f"\n  {'id':>7}  {'token':<30}{'norm':>10}{'x mean':>9}")
    for tid, name in sorted(LLAMA3_SPECIALS.items()):
        if tid >= norms.shape[0]:
            continue
        value = norms[tid].item()
        mark = "  <- eos" if tid == eos else ""
        print(f"  {tid:>7}  {name:<30}{value:>10.4f}{value / mean:>9.3f}{mark}")

    ratio = norms[eos].item() / mean
    print()
    if ratio < 0.5:
        print(f"  REFUSE: the stop token's row is {ratio:.2f}x an ordinary "
              "token's. The head is frozen under LoRA, so this token cannot be "
              "learned — pick the one this checkpoint was actually trained to "
              "end with.")
        return 1
    print(f"  OK: the stop token's row is {ratio:.2f}x an ordinary token's, so "
          "the frozen head can produce it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
