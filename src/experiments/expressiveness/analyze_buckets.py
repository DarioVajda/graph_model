"""
Analyse the L (packed token length) and N (node count) distributions for the
train-mode node ranges, to choose tight flex `len_buckets` / `node_buckets`.

L and N depend only on the node count and the label tokenisation — NOT on graph
structure — so this skips graph/feature construction entirely and is exact:

    N = size + 1                       # +1 for the prompt node
    L = sum(token_len(label) for each node) + token_len(prompt text)

with the same `add_special_tokens=False`, no-EOS tokenisation the real pipeline
uses (`TextGraphDataset.tokenize`), and the same 0.8x–1.2x `num_nodes` range that
`RunConfig.__post_init__` derives.

Run: python3 -m src.experiments.expressiveness.analyze_buckets
"""

import random
import statistics

from transformers import AutoTokenizer

from .config import MODEL_NAME
from .data_gen import _spreadsheet_label
from ...models.flex_kernel import default_len_buckets, default_node_buckets, bucketize

TARGETS = [100, 200, 500, 1000, 2000]
SAMPLES = 3000          # graphs sampled per size (cheap — pure tokenisation)
BATCH_SIZES = [2, 4]    # bench / train effective per-step batch
SEED = 0


def _pct(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _stats_line(name, xs):
    return (f"  {name}: min={min(xs):4d}  p1={_pct(xs,1):6.0f}  p25={_pct(xs,25):6.0f}  "
            f"p50={_pct(xs,50):6.0f}  p75={_pct(xs,75):6.0f}  p99={_pct(xs,99):6.0f}  "
            f"max={max(xs):4d}  mean={statistics.mean(xs):6.1f}")


def sample_LN(tok, target, samples=SAMPLES):
    """Return (Ns, Ls) for `samples` graphs drawn from the 0.8x–1.2x range."""
    mn, mx = round(0.8 * target), round(1.2 * target)
    pool_size = max(26, mx)
    pool = [" " + _spreadsheet_label(i) for i in range(pool_size)]
    pool_len = [len(tok(lbl, add_special_tokens=False).input_ids) for lbl in pool]

    Ns, Ls = [], []
    for _ in range(samples):
        size = random.randint(mn, mx)
        idx = random.sample(range(pool_size), size)
        node_tokens = sum(pool_len[i] for i in idx)
        x, y = random.sample(idx, 2)
        ans = random.choice(["Yes", "No"])
        prompt = f"Are the nodes{pool[x]} and{pool[y]} connected? {ans}"
        Ls.append(node_tokens + len(tok(prompt, add_special_tokens=False).input_ids))
        Ns.append(size + 1)
    return Ns, Ls


def batched_buckets(Ns, Ls, batch_size, len_spec, node_spec):
    """Group consecutive samples into batches, map each batch's (max L, max N) to
    its buckets, and report the realised combos + padding waste.

    Returns (n_combos, len_waste_frac, node_pad_area_frac, combo_counts)."""
    combos = {}
    raw_len = pad_len = 0
    raw_area = pad_area = 0
    for i in range(0, len(Ns) - batch_size + 1, batch_size):
        bN = max(Ns[i:i + batch_size]); bL = max(Ls[i:i + batch_size])
        tL = bucketize(bL, len_spec); tN = bucketize(bN, node_spec)
        combos[(tL, tN)] = combos.get((tL, tN), 0) + 1
        raw_len += bL; pad_len += tL
        raw_area += bN * bN; pad_area += tN * tN   # (N,N) bias area is the memory driver
    return (len(combos),
            1.0 - raw_len / pad_len,
            1.0 - raw_area / pad_area,
            dict(sorted(combos.items())))


def evaluate(Ns, Ls, len_spec, node_spec, label):
    print(f"\n  [{label}]  len_buckets={len_spec}  node_buckets={node_spec}")
    for bs in BATCH_SIZES:
        n, lw, aw, combos = batched_buckets(Ns, Ls, bs, len_spec, node_spec)
        print(f"    batch={bs}: {n:2d} combos | L-pad waste {lw:5.1%} | "
              f"(N,N)-area pad {aw:5.1%}")
        print(f"             combos(count): " +
              ", ".join(f"{k}:{v}" for k, v in combos.items()))


# Recommended tight ladders per size: up to 4 L buckets + 2 N buckets (mean, max),
# all multiples of 128, top buckets = hard upper bounds so bucketize never raises
# mid-training. (L caps below 4 entries at small sizes — the 128 grid spans <4
# blocks there.) Defaults are shown alongside for contrast.
CANDIDATES = {
    100:  ([128, 256],              [128]),
    200:  ([256, 384],              [256]),
    500:  ([640, 768],              [512, 640]),
    1000: ([1408, 1664, 1792],      [1024, 1280]),
    2000: ([3200, 3712, 4096, 4224],[2048, 2432]),
}


def main():
    random.seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    for target in TARGETS:
        Ns, Ls = sample_LN(tok, target)
        mn, mx = round(0.8 * target), round(1.2 * target)
        print("=" * 92)
        print(f"num_nodes={target}  (range {mn}-{mx}, +1 prompt node -> N in {mn+1}-{mx+1})  "
              f"| {SAMPLES} samples")
        print("=" * 92)
        print(_stats_line("N", Ns))
        print(_stats_line("L", Ls))
        print(f"  L/N ratio: mean={statistics.mean(l/n for l, n in zip(Ls, Ns)):.3f}")
        evaluate(Ns, Ls, default_len_buckets, default_node_buckets, "collator defaults")
        if target in CANDIDATES:
            lb, nb = CANDIDATES[target]
            evaluate(Ns, Ls, lb, nb, "proposed tight")


if __name__ == "__main__":
    main()
