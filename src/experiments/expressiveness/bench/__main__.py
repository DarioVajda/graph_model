"""
Standalone benchmark entry point (kept outside the JSON sweep wrapper).

Bench is a throughput/memory probe, not an accuracy sweep, so it keeps a plain
argparse interface and sweeps multiple impls / k-hops in one process to build a
comparison table. Run one fixed graph size per invocation::

    python3 -m src.experiments.expressiveness.bench \
        --impls v2-eager,v2-flex --k-hops 0,1 --num-nodes 1000

Results append to ``results/benchmarks.jsonl`` (see :data:`bench.BENCH_RESULTS_PATH`).
"""

import argparse

from ..config import RunConfig
from .bench import run_bench_mode


def _int_list(raw):
    return [int(s) for s in raw.split(",") if s.strip()]


def _str_list(raw):
    return [s.strip() for s in raw.split(",") if s.strip()]


def build_parser():
    d = RunConfig()
    p = argparse.ArgumentParser(
        prog="python3 -m src.experiments.expressiveness.bench",
        description="Large-graph throughput + peak-memory + sparsity probe for the GTLM graph bias.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--impls", type=_str_list, default=list(d.impls),
                   help="Comma-separated implementations: v0-eager,v2-eager,v2-flex.")
    p.add_argument("--k-hops", type=_int_list, default=list(d.k_hops),
                   help="Comma-separated k-hop radii to sweep (v2 only).")
    p.add_argument("--k-hop-directed", action="store_true", default=d.k_hop_directed)
    p.add_argument("--difficulty", choices=("HARD", "EASY"), default=d.difficulty)
    p.add_argument("--num-nodes", type=int, default=d.num_nodes,
                   help="Fixed graph size to benchmark.")
    p.add_argument("--model-name", default=d.model_name)
    p.add_argument("--magnetic-m", type=int, default=d.magnetic_m)
    p.add_argument("--flex-compile-mode", default=d.flex_compile_mode)
    p.add_argument("--ordering", choices=("rcm", "original"), default=d.ordering)
    p.add_argument("--len-buckets", type=_int_list, default=d.len_buckets)
    p.add_argument("--node-buckets", type=_int_list, default=d.node_buckets)
    p.add_argument("--bench-batch-size", type=int, default=d.bench_batch_size)
    p.add_argument("--bench-num-warmup", type=int, default=d.bench_num_warmup)
    p.add_argument("--bench-num-iters", type=int, default=d.bench_num_iters)
    p.add_argument("--bench-num-examples", type=int, default=d.bench_num_examples)
    p.add_argument("--density-sample-graphs", type=int, default=d.density_sample_graphs)
    p.add_argument("--density-sample-batches", type=int, default=d.density_sample_batches)
    return p


def config_from_args(args):
    return RunConfig(
        mode="bench",
        impls=tuple(args.impls),
        k_hops=tuple(args.k_hops),
        k_hop_directed=args.k_hop_directed,
        difficulty=args.difficulty,
        num_nodes=args.num_nodes,
        model_name=args.model_name,
        magnetic_m=args.magnetic_m,
        flex_compile_mode=args.flex_compile_mode,
        ordering=args.ordering,
        len_buckets=tuple(args.len_buckets) if args.len_buckets else None,
        node_buckets=tuple(args.node_buckets) if args.node_buckets else None,
        bench_batch_size=args.bench_batch_size,
        bench_num_warmup=args.bench_num_warmup,
        bench_num_iters=args.bench_num_iters,
        bench_num_examples=args.bench_num_examples,
        density_sample_graphs=args.density_sample_graphs,
        density_sample_batches=args.density_sample_batches,
    ).validate()


if __name__ == "__main__":
    run_bench_mode(config_from_args(build_parser().parse_args()))
