#!/usr/bin/env bash
# =============================================================================
# sbatch_chain_decoy.sh — the DECOY-REFERENCE build (fan_out=2).
# =============================================================================
# Why this exists. At fan_out=1 every content node has exactly one successor, so
# the content subgraph is FUNCTIONAL and the answer is the unique node at
# SPD == hops from the start. Measured on the built data (40 graphs x k=1..4):
# out-degree {1: 880}, SPD(start->answer) == k in every graph, exactly one node at
# that distance. The graph arm therefore never traverses anything — it locates the
# start by name and reads the answer off the distance bias. That is O(1) in k,
# against the flat arm's O(k) text walk, and it is why the learnability control
# came out graph 1.000/1.000/1.000/1.000 vs flat 1.000/0.995/0.145/0.135: the
# graph arm's accuracy is FLAT in k because k costs it nothing.
#
# So the control measured retrieval-with-a-shortcut, not multi-hop reasoning.
# fan_out=2 removes the shortcut: each node emits 1 real "Continue at" pointer and
# 1 explicitly-labelled "Decoy reference", both entering the DiGraph identically.
# ~2**k nodes then sit at distance k, so topology can PRUNE the candidate set but
# cannot identify the answer; only the text says which reference is real. The
# decoy is named as a decoy on purpose — the aim is not to make the task ambiguous
# but to move the disambiguating signal out of the topology into the text, where
# BOTH arms can read it.
#
# N=32 ONLY. The first build used 011/012's cell (N=16,32) to match the control
# exactly, and the Phase 2 gate rejected N=16 — on a statistic that was WRONG.
#
# CORRECTED 2026-07-30. The gate printed "answer ALONE at distance k" (N=16/k=4:
# 36%) but computed `len(shell_k) == 1`, i.e. how often the distance-k shell is a
# SINGLETON — never whether that singleton is the answer. At N=16/k=4 the shell is
# a singleton 34.5% of the time and the singleton is the answer only 9% of the
# time. Remeasured on 200 test graphs/cell with an oracle that sees every distance
# and no text, playing the best distance-only strategy available:
#
#     cell       P(answer in shell_k)  mean|shell_k|  SPD-only  chance    leak
#     N=16 k=3          59.0%              3.20        19.0%     7.1%  +11.8pp
#     N=32 k=3          81.5%              5.34        15.6%     3.3%  +12.2pp
#     N=16 k=4          27.0%              1.79        16.2%     7.1%   +9.1pp
#     N=32 k=4          52.5%              5.63         9.9%     3.3%   +6.5pp
#
# Excess over chance is comparable at both cells, so N=16 does NOT leak more than
# N=32. Small N costs SIGNAL, not fairness: at N=16/k=4 a decoy shortcuts the
# answer to below distance k in 73% of graphs, leaving the structural channel
# mostly noise, so a graph~=flat result there would be uninterpretable. N=32 keeps
# the answer inside the shell for 52.5% of k=4 graphs with ~5.6 candidates of 30 —
# structure prunes, but never to one, and usually still contains the answer.
#
# N=32 therefore remains the right cell for this build, for the signal reason
# rather than the leak reason. N=16 is admissible in a future sweep provided the
# per-cell shell statistics are reported alongside the accuracies.
#
# Both fan_out=1 and fan_out=2 are built here at N=32 so the control is matched on
# the cell as well as everything else; the pre-existing fan_out=1 data is N=16,32
# and would leave cell size confounded with fan-out.
#
# A high "shortcut below k" rate at N=32/k=4 (48%) is NOT a defect: it means a
# decoy reached the answer in fewer than k steps, so distance is uninformative
# rather than identifying. The model cannot exploit it because it does not know
# the shortened distance.
#
# Decoys are deliberately NOT kept away from the gold chain. Doing so would hold
# SPD(start->answer) at exactly k, but it would also leave chain nodes as the only
# ones with in-degree 1 — a fresh structural shortcut. Decoys point anywhere and
# the shortcut rate is measured instead (diag_decoy.py, the Phase 2 gate).
#
# Needle cost measured, not assumed: 18 -> 27 tokens at fan_out=2, leaving 29 of
# T=64 for filler after SUFFIX_SLACK. Fits.
#
# MEMORY: 128G. The comparable small-cell fan_out=1 builds peaked at ~38 GB; the
# N=64 cell is what drives memory (61.7 GB at n_train=2000) and this is N<=32.
# A 96 G request OOM-killed the N=64 8k builds (119549-119552), so no economising.
#
# CPU-only: the build is HF `datasets.map` plumbing, not the eigendecomposition.
# Do NOT use the `amd` partition (MI210/ROCm cannot start this CUDA container) --
# sbatch prints a hint suggesting it for CPU-only jobs; ignore it.
#
# Idempotent: a split already on disk is skipped, so re-running resumes.
#
#   ./src/experiments/context/sbatch_chain_decoy.sh [PARTITION]
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

PARTITION="${1:-frida}"
CONTAINER="/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh"
mkdir -p "$REPO/job_logs"

# Must match 017/018 — these knobs form the cache key.
HOPS=(1 2 3 4)
FANS=(1 2)
COMMON="--mode data_prep --node-counts 32 --token-counts 64 --max-train-len 16384 \
--n-train 8000 --n-dev 200 --n-test 200 --code-len 3 --id-pool 4096 --data-seed 42 \
--model-name meta-llama/Llama-3.2-1B --spd --max-spd 8 --no-rrwp --magnetic \
--magnetic-dim 128 --magnetic-m 128 --magnetic-q 0.25"

STAMP="$(date +%Y%m%d_%H%M%S)"
for f in "${FANS[@]}"; do
for k in "${HOPS[@]}"; do
  TAG="h${k}f${f}"
  SCRIPT="$REPO/job_logs/ctx_decoy_${TAG}_${STAMP}.sh"
  LOG="$REPO/job_logs/ctx_decoy_${TAG}_${STAMP}_%j.out"
  {
    echo "#!/usr/bin/env bash"
    echo "set -x"
    echo "python -m src.experiments.context $COMMON --hops $k --fan-out $f"
  } > "$SCRIPT"
  chmod +x "$SCRIPT"

  WRAP="srun --container-image=$CONTAINER --container-mounts=/shared:/shared \
env HOME=$HOME PYTHONUNBUFFERED=1 SWEEP_PROJECT_ROOT=$REPO SWEEP_VENV_BIN=$REPO/.venv/bin \
bash $REPO/sweep/slurm_launch.sh ctx_decoy_${TAG}_${STAMP} $SCRIPT"

  echo "[submit] hops=$k fan_out=$f -> $LOG"
  sbatch -p "$PARTITION" -A povejmo -c 16 --mem 128G -t 04:00:00 \
         -J "ctx_decoy_${TAG}" -o "$LOG" --wrap "$WRAP"
done
done
