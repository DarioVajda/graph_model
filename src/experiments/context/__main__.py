"""
Context experiment — standalone single-run entry point.

Three modes, run in this order:

    python3 -m src.experiments.context --mode data_prep         # build the .gtds caches
    python3 -m src.experiments.context --mode train             # train on the capped mixture
    python3 -m src.experiments.context --mode grid --checkpoint-path ./checkpoints/...

The generic ``sweep`` runner invokes this once per resolved config, rendering each
config key to the matching flag:

    python3 -m sweep src.experiments.context src/experiments/context/configs/003_train_16k.jsonc
"""

import argparse
import os

from .config import MODES, RunConfig

CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


TEMPLATE = """\
{
  // ─────────────────────────────────────────────────────────────────────────
  // Context-exhaustion sweep config (JSONC: // comments + trailing commas ok).
  //   python3 -m sweep src.experiments.context <this file>
  // ─────────────────────────────────────────────────────────────────────────

  "name": "my_sweep",
  "results_dir": "src/experiments/context/results",

  "execution": {
    "mode": "local"                   // "local" | "sbatch"
  },

  "mode": "train",
  "seed": [0, 1, 2],

  "model_name": "meta-llama/Llama-3.2-1B",
  // The grid axes are STRINGS, not JSON lists: the sweep runner reads a list as a
  // sweep AXIS, so [8, 16] would expand into two runs with one-value grids.
  "node_counts": "8,16,32,64,128",
  "token_counts": "32,64,128,256,512",
  "max_train_len": 16384,
  "n_train": 4000,

  "num_epochs": 3,
  "batch_size": 1,
  "accumulation_steps": 8,
  "lr": 1e-4,
  "bias_lr": 5e-3,
  "lora_r": 64,
  "k_hop": 0,
  "graph_attn_impl": "flex",
  "dtype": "bf16",
  "wandb_project": null
}
"""


def _int_list(raw):
    """Comma-splitting type for the grid axes (rule 3 of the sweep contract)."""
    if isinstance(raw, (list, tuple)):
        return tuple(int(v) for v in raw)
    return tuple(int(s) for s in str(raw).split(",") if s.strip())


def _str_list(raw):
    return [s.strip() for s in raw.split(",") if s.strip()]


def build_parser():
    d = RunConfig()
    p = argparse.ArgumentParser(
        prog="python3 -m src.experiments.context",
        description="Run ONE context-exhaustion configuration (data_prep | train | grid).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    B = argparse.BooleanOptionalAction

    p.add_argument("--init", nargs="?", const="context", default=None, metavar="NAME",
                   help="Write a sweep-config template to configs/<NAME>.jsonc and exit.")
    # Choices come from config.MODES so adding a mode there cannot leave the CLI
    # rejecting it (a duplicated literal here shipped flat_grid to the cluster
    # and failed 44 s in, sweep job 119421).
    p.add_argument("--mode", choices=MODES, default=d.mode)

    # ── model + LoRA ───────────────────────────────────────────────────────────
    p.add_argument("--model-name", default=d.model_name)
    p.add_argument("--lora", action=B, default=d.lora)
    p.add_argument("--lora-r", type=int, default=d.lora_r)
    p.add_argument("--lora-alpha", type=int, default=d.lora_alpha)
    p.add_argument("--lora-dropout", type=float, default=d.lora_dropout)
    p.add_argument("--active-params", type=_str_list, default=list(d.active_params))

    # ── graph-bias features ────────────────────────────────────────────────────
    p.add_argument("--spd", action=B, default=d.spd)
    p.add_argument("--max-spd", type=int, default=d.max_spd)
    p.add_argument("--rrwp", action=B, default=d.rrwp)
    p.add_argument("--max-rw-steps", type=int, default=d.max_rw_steps)
    p.add_argument("--magnetic", action=B, default=d.magnetic)
    p.add_argument("--magnetic-groups", type=int, default=d.magnetic_groups,
                   help="Layer-sharing granularity of the magnetic bias: G instances "
                        "instead of one per layer (0 = per-layer, 1 = one for the "
                        "whole stack). Model-side only; the dataset is unchanged.")
    p.add_argument("--magnetic-dim", type=int, default=d.magnetic_dim)
    p.add_argument("--magnetic-q", type=float, default=d.magnetic_q)
    p.add_argument("--magnetic-m", type=int, default=d.magnetic_m)

    # ── attention ──────────────────────────────────────────────────────────────
    p.add_argument("--k-hop", type=int, default=d.k_hop)
    p.add_argument("--k-hop-directed", action=B, default=d.k_hop_directed)
    p.add_argument("--graph-attn-impl", choices=("flex", "eager"), default=d.graph_attn_impl)
    p.add_argument("--dtype", choices=("bf16", "fp32"), default=d.dtype)
    p.add_argument("--compile-mode", default=d.compile_mode)

    # ── grid + dataset ─────────────────────────────────────────────────────────
    p.add_argument("--node-counts", type=_int_list, default=d.node_counts,
                   help="Comma-separated N values (total nodes incl. QUESTION + PROMPT).")
    p.add_argument("--token-counts", type=_int_list, default=d.token_counts,
                   help="Comma-separated T values (tokens per content node, exact).")
    p.add_argument("--max-train-len", type=int, default=d.max_train_len,
                   help="Cells longer than this are excluded from TRAINING (still evaluated).")
    p.add_argument("--n-train", type=int, default=d.n_train)
    p.add_argument("--n-dev", type=int, default=d.n_dev)
    p.add_argument("--n-test", type=int, default=d.n_test, help="Graphs per grid cell.")
    p.add_argument("--hops", type=int, default=d.hops,
                   help="0 = lookup task; k>0 = k-hop pointer chasing (README §A.1)")
    p.add_argument("--hop-counts", type=_int_list, default=d.hop_counts,
                   help="Comma-separated k values to MIX (the main sweep). Empty = use "
                        "the scalar --hops for every graph. When set, k is drawn per "
                        "graph and the QUESTION states the drawn k, so the model must "
                        "read the task off the input rather than assume a constant.")
    p.add_argument("--fan-out", type=int, default=d.fan_out,
                   help="Out-edges per content node when hops>0: 1 real pointer plus "
                        "fan_out-1 labelled decoy references. 1 = the original "
                        "functional-graph build, where SPD alone identifies the answer.")
    p.add_argument("--train-shards", type=int, default=d.train_shards,
                   help="Split the train build into this many contiguous blueprint "
                        "ranges, each buildable as its own job (bounds peak RAM).")
    p.add_argument("--train-shard", type=int, default=d.train_shard,
                   help="Build ONLY this shard index and nothing else. Omit to build "
                        "dev + the test grid instead. See --mode data_merge.")
    p.add_argument("--only-cells", type=str, default=d.only_cells,
                   help="Build (or, in --mode grid, score) only these test-grid cells, as "
                        "NxT comma-joined (e.g. 128x512,64x256). Empty = all. Does not "
                        "affect the cache key.")
    p.add_argument("--only-hops", type=str, default=d.only_hops,
                   help="In --mode grid, score only these k values, comma-joined (e.g. 1,3). "
                        "Empty = every k the build contains. Does not affect the cache key.")
    p.add_argument("--code-len", type=int, default=d.code_len)
    p.add_argument("--id-pool", type=int, default=d.id_pool)
    p.add_argument("--corpus-tokens", type=int, default=d.corpus_tokens)
    p.add_argument("--data-seed", type=int, default=d.data_seed)
    p.add_argument("--data-format-version", type=int, default=d.data_format_version)

    # ── training schedule ──────────────────────────────────────────────────────
    p.add_argument("--num-epochs", type=int, default=d.num_epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--accumulation-steps", type=int, default=d.accumulation_steps)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--bias-lr", type=float, default=d.bias_lr)
    p.add_argument("--bias-weight-decay", type=float, default=d.bias_weight_decay)
    p.add_argument("--eval-steps", type=int, default=d.eval_steps)
    p.add_argument("--max-steps", type=int, default=d.max_steps)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--gradient-checkpointing", action=B, default=d.gradient_checkpointing)

    # ── grid mode ──────────────────────────────────────────────────────────────
    p.add_argument("--checkpoint-path", default=d.checkpoint_path,
                   help="(grid) checkpoint directory to score.")

    p.add_argument("--wandb-project", default=d.wandb_project)

    # ── sweep-runner bookkeeping ───────────────────────────────────────────────
    p.add_argument("--runs-jsonl", default=None, help="(runner) JSONL to append this run's record to.")
    p.add_argument("--run-name", default=None, help="(runner) this run's name within the sweep.")
    p.add_argument("--sweep-id", default=None, help="(runner) the sweep this run belongs to.")
    p.add_argument("--resume", action="store_true", help="Resume training from the output dir.")
    return p


def config_from_args(args):
    """Build (and validate) a RunConfig from parsed args."""
    return RunConfig(
        mode=args.mode,
        model_name=args.model_name,
        lora=args.lora, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, active_params=tuple(args.active_params),
        spd=args.spd, max_spd=args.max_spd,
        rrwp=args.rrwp, max_rw_steps=args.max_rw_steps,
        magnetic=args.magnetic, magnetic_groups=args.magnetic_groups,
        magnetic_dim=args.magnetic_dim,
        magnetic_q=args.magnetic_q, magnetic_m=args.magnetic_m,
        k_hop=args.k_hop, k_hop_directed=args.k_hop_directed,
        graph_attn_impl=args.graph_attn_impl, dtype=args.dtype,
        compile_mode=args.compile_mode,
        node_counts=tuple(args.node_counts), token_counts=tuple(args.token_counts),
        max_train_len=args.max_train_len,
        n_train=args.n_train, n_dev=args.n_dev, n_test=args.n_test,
        hops=args.hops, hop_counts=args.hop_counts, fan_out=args.fan_out,
        code_len=args.code_len, id_pool=args.id_pool, corpus_tokens=args.corpus_tokens,
        train_shards=args.train_shards, train_shard=args.train_shard,
        only_cells=args.only_cells, only_hops=args.only_hops,
        data_seed=args.data_seed, data_format_version=args.data_format_version,
        num_epochs=args.num_epochs, batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        lr=args.lr, bias_lr=args.bias_lr, bias_weight_decay=args.bias_weight_decay,
        eval_steps=args.eval_steps, max_steps=args.max_steps, seed=args.seed,
        num_workers=args.num_workers, gradient_checkpointing=args.gradient_checkpointing,
        checkpoint_path=args.checkpoint_path,
        wandb_project=args.wandb_project,
    ).validate()


def _do_init(name):
    if not (name.endswith(".json") or name.endswith(".jsonc")):
        name += ".jsonc"
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    path = os.path.join(CONFIGS_DIR, name)
    with open(path, "w") as f:
        f.write(TEMPLATE)
    print(f"Wrote sweep template to {path}\n"
          f"Edit it, then run:  python3 -m sweep src.experiments.context {path}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.init is not None:
        _do_init(args.init)
        return 0

    cfg = config_from_args(args)

    if cfg.mode == "data_prep":
        from .process_dataset import run_data_prep_mode
        run_data_prep_mode(cfg)
        return 0

    if cfg.mode == "data_merge":
        from .process_dataset import run_data_merge_mode
        run_data_merge_mode(cfg)
        return 0

    if cfg.mode == "grid":
        from .grid import run_grid_mode
        run_grid_mode(cfg, runs_jsonl=args.runs_jsonl, run_name=args.run_name,
                      sweep_id=args.sweep_id)
        return 0

    if cfg.mode == "flat_grid":
        from .flat import run_flat_grid_mode
        run_flat_grid_mode(cfg, runs_jsonl=args.runs_jsonl, run_name=args.run_name,
                           sweep_id=args.sweep_id)
        return 0

    if cfg.mode == "flat_train":
        from .flat import run_flat_train_mode
        run_flat_train_mode(cfg, runs_jsonl=args.runs_jsonl, run_name=args.run_name,
                            sweep_id=args.sweep_id, resume=args.resume)
        return 0

    from .train import run_train_mode
    run_train_mode(cfg, runs_jsonl=args.runs_jsonl, run_name=args.run_name,
                   sweep_id=args.sweep_id, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
