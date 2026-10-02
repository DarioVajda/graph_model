#!/usr/bin/env bash
# =============================================================================
# assistant/pipeline/run.sh — build the §9.4 assistant set, end to end.
# =============================================================================
# Six stages in the order §9.4 fixes, because a change to an earlier one
# invalidates everything written after it:
#
#   build   declare the intents and render their replies          (CPU)
#   ask     write the person's turn from the situation and the ask (GPU)
#   voice   re-voice the rendered reply under the style brief      (GPU)
#   judge   responsive / preserved / added, greedy                 (GPU)
#   accept  the three checks, the held-out filter, dedup, ceilings (CPU)
#   topup   build again for whatever accept did not deliver        (CPU+GPU)
#   compose few-shot demonstrations drawn from accepted rows       (CPU)
#
# **`--n-train` and `--n-test` are accepted rows, not intents written.** About a
# quarter of a build does not survive `accept`, so asking for 11,000 and taking
# what came out the far end delivered 8,009 — and the fix for that was a top-up
# build merged in by hand, twice. The first `build` now oversamples by `YIELD`
# and `topup` closes whatever gap is left, so the pipeline delivers the number it
# was asked for or says why it could not.
#
# Each stage is a separate blocking sbatch, so a stage that fails stops the run
# with its own log rather than poisoning the next one. Re-running with the same
# --out skips nothing: stages are cheap to redo and a half-written stage is the
# one thing worth never trusting.
#
#   src/generalist/assistant/pipeline/run.sh --out src/generalist/results/assistant/v5 \
#       --n-train 11000 --n-test 900
#   src/generalist/assistant/pipeline/run.sh --out .../v5 --from voice
#
# `--domain` names the assistant domain (`assistant/domain.py`) and goes to every
# stage; it defaults to molecules, the only one registered so far.
#
# `--from` restarts at a named stage. The writer stages need a card that fits a
# 31B model in bf16, which is why GPU_CONSTRAINT is set here and not left to the
# launcher's default — that default admits the 40G A100s, and a 31B writer does
# not fit on one.
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO"

OUT=""
CONFIG="src/generalist/configs/probes/008_molecule_generalist_instruct.jsonc"
CELL="molecule_generalist_instruct_graph_s0"
N_TRAIN=11000
N_TEST=900
FROM="build"
SEED=0
DOMAIN="molecules"
MODEL="/shared/workspace/povejmo/huggingface_cache/hub/models--google--gemma-4-31B-it/snapshots/b9ea41a2887d8607f594846523f94c6cc75ac8a4"

while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --cell) CELL="$2"; shift 2 ;;
    --n-train) N_TRAIN="$2"; shift 2 ;;   # accepted rows, not intents written
    --n-test) N_TEST="$2"; shift 2 ;;     # ditto
    --from) FROM="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --domain) DOMAIN="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$OUT" ] || { echo "--out is required" >&2; exit 2; }

# `.venv_writer` and `nemo_26.04.sqsh` travel together: the venv's python is a
# symlink into that image and exists nowhere else.
#
# These go through `env` and not as a prefix: bash only reads `VAR=value` as an
# assignment when it is a literal word, so an array expanded in front of a
# command is executed as the command, and the whole stage dies on
# "VENV_BIN=.venv_writer/bin: No such file or directory".
WRITER_ENV=(VENV_BIN=.venv_writer/bin
            CONTAINER=/shared/workspace/povejmo/containers/nemo_26.04.sqsh
            GPU=1 GPU_CONSTRAINT='GPU_BRD:B200|GPU_BRD:B300' MEM=200G
            TIME=06:00:00)

# SHARDS fans each writer stage out over that many single-GPU jobs. All three
# take `--shard/--shards` and split by batch file, and each writes one output
# file per input batch, so the shards never collide and the accept pass reads
# the union without knowing it was sharded. One shard holds one copy of the 31B
# writer, so the width is bounded by free cards rather than by memory: a node
# carries 2 TB against 8 x 200 G. Measured single-shard on ~595 batches: ask
# 71 min, voice 48, judge 51 — so the model load, a couple of minutes, is what
# the width stops paying for somewhere above ten.
SHARDS="${SHARDS:-1}"

# How much of a build survives `accept`, used to size the first one. 0.74 is what
# the 2026-09-20 rebuild measured over 11,900 intents; it is a starting point, and
# `topup` corrects whatever it gets wrong rather than depending on it.
#
# `MIN_YIELD` is the floor a round may assume, and it is well under 0.74 on
# purpose: a top-up deduplicates against everything already accepted, so it
# converts worse than the first build did, and assuming otherwise is how a round
# comes back still short. Overshooting a round costs one build of CPU; undershooting
# costs a whole GPU round trip.
YIELD="${YIELD:-0.74}"
MIN_YIELD="${MIN_YIELD:-0.40}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"

# Rows to be within of the target before it counts as delivered. Chasing the last
# handful is a GPU round trip for a tenth of a per cent of the set.
TOLERANCE="${TOLERANCE:-0.01}"

# Intents to write for `want` accepted rows at yield `y`, floored at MIN_YIELD.
build_n() {
  awk -v t="$1" -v y="$2" -v m="$MIN_YIELD" \
      'BEGIN { if (y + 0 < m + 0) y = m; printf "%d", int(t / y + 0.999) }'
}

# Fan one writer stage out and fail the pipeline if any shard fails. The `|| rc=1`
# is load-bearing under `set -e`: a bare `wait` on a failed shard would leave the
# later stages to run against a half-written directory, which is the one thing
# this pipeline promises not to do.
run_sharded() {
  local name="$1"; shift
  local pids=() rc=0 s
  for ((s = 0; s < SHARDS; s++)); do
    env NAME="${name}_s${s}" "${WRITER_ENV[@]}" \
      bash src/generalist/tools/run_py.sh "$@" --shard "$s" --shards "$SHARDS" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
  [ "$rc" = 0 ] || { echo "$name: a shard failed" >&2; exit 1; }
}

# The three writer passes and the accept pass, as functions, because `topup` runs
# them again over one round's batches. `$1` is the suffix that keeps a round's job
# names apart in the queue and `$2` the glob that keeps it off the batches already
# written — every pass writes one output per input batch, named after it, so a
# round lands beside the earlier ones instead of on top of them.
pass_ask() {
  run_sharded "intent_ask$1" -m src.generalist.assistant.pipeline.write \
      --model "$MODEL" --batches "$OUT" --out "$OUT/ask" --pass ask \
      --writer gemma-4-31B-it --glob "$2" --domain "$DOMAIN"
}

pass_voice() {
  run_sharded "intent_voice$1" -m src.generalist.assistant.pipeline.write \
      --model "$MODEL" --batches "$OUT" --out "$OUT/voice" --pass voice \
      --asks "$OUT/ask" --writer gemma-4-31B-it --glob "$2" --domain "$DOMAIN"
}

pass_judge() {
  run_sharded "intent_judge$1" -m src.generalist.assistant.pipeline.judge \
      --model "$MODEL" --batches "$OUT" --asks "$OUT/ask" \
      --voiced "$OUT/voice" --out "$OUT/judged" --glob "$2" --domain "$DOMAIN"
}

# Always over the whole directory: the dedup pool has to be the whole set, or each
# round deduplicates against itself and the union carries the pairs between them.
run_accept() {
  NAME=intent_accept CPUS=8 MEM=32G TIME=01:00:00 \
    bash src/generalist/tools/run_py.sh -m src.generalist.assistant.pipeline.accept \
      --batches "$OUT" --asks "$OUT/ask" --voiced "$OUT/voice" \
      --judged "$OUT/judged" --out "$OUT/accepted" --seed "$SEED" \
      --domain "$DOMAIN"
}

accepted_field() { jq -r "$1" "$OUT/accepted/summary.json"; }

STAGES=(build ask voice judge accept topup compose)
started=0
should_run() {
  if [ "$1" = "$FROM" ]; then started=1; fi
  [ "$started" = 1 ]
}
# A --from naming no stage is a typo, and a typo that silently runs everything
# is worse than one that runs nothing.
case " ${STAGES[*]} " in *" $FROM "*) ;; *)
  echo "--from must be one of: ${STAGES[*]}" >&2; exit 2 ;;
esac

if should_run build; then
  echo "=== build ==="
  echo "targets: $N_TRAIN train, $N_TEST test accepted rows (yield $YIELD)"
  NAME=intent_build CPUS=8 MEM=96G TIME=03:00:00 \
    bash src/generalist/tools/run_py.sh -m src.generalist.assistant.pipeline.build \
      --config "$CONFIG" --cell "$CELL" --out "$OUT" \
      --n-train "$(build_n "$N_TRAIN" "$YIELD")" \
      --n-test "$(build_n "$N_TEST" "$YIELD")" --seed "$SEED" \
      --domain "$DOMAIN"
fi

if should_run ask; then
  echo "=== ask ==="
  pass_ask "" "*batch-*.json"
fi

if should_run voice; then
  echo "=== voice ==="
  pass_voice "" "*batch-*.json"
fi

if should_run judge; then
  echo "=== judge ==="
  pass_judge "" "*batch-*.json"
fi

if should_run accept; then
  echo "=== accept ==="
  # --judged is on because the judge can only remove rows here, so its errors
  # cost yield and cannot cost correctness. On the final build that price was
  # measured by reading all 104 of its refusals: precision 0.231, so roughly 80
  # correct rows in 14,900. Two of the three misreadings behind that are fixed in
  # the judge's system prompt (molecules/prompts.py) and its per-row prompt; the number is stale the moment they
  # take effect, so read the refusals again rather than quoting 0.231 forward.
  # --dedup-max takes accept.DEDUP_MAX, which is the settled 0.85.
  run_accept
fi

if should_run topup; then
  echo "=== topup ==="
  # Each round is a whole build of its own, merged in rather than appended to:
  # `--id-prefix` puts a letter on the ids *and* the batch filenames, because ids
  # restart at `train-00000` every build and the ask, voice and judge passes join
  # on the id. Without it a round reads one row's reply against another row's
  # statements, silently. The seed moves with the round for the same reason a
  # prefix does — the same seed redraws the same molecules in the same order, so
  # a top-up at `--seed $SEED` would be almost entirely duplicates.
  rounds=(b c d e f)
  # The *marginal* yield, which is the one that sizes a round. The cumulative
  # figure in `summary.json` is close to the first build's and barely moves as
  # rounds are added, so sizing off it undershoots by the same margin every time
  # and the loop converges one slow round at a time. After a round this becomes
  # what that round actually converted. The first round has nothing better than
  # the cumulative figure to go on, and `MIN_YIELD` is the floor under it.
  yield="$(accepted_field '.yield // 0.74')"
  for ((round = 1; round <= MAX_ROUNDS; round++)); do
    have_train="$(accepted_field '.by_role.train // 0')"
    have_test="$(accepted_field '.by_role.test // 0')"
    had="$(accepted_field '.accepted // 0')"

    short_train=$(( N_TRAIN - have_train ))
    short_test=$(( N_TEST - have_test ))
    if [ "$short_train" -lt 0 ]; then short_train=0; fi
    if [ "$short_test" -lt 0 ]; then short_test=0; fi
    tol_train="$(awk -v n="$N_TRAIN" -v t="$TOLERANCE" 'BEGIN{printf "%d", n*t}')"
    tol_test="$(awk -v n="$N_TEST" -v t="$TOLERANCE" 'BEGIN{printf "%d", n*t}')"

    if [ "$short_train" -le "$tol_train" ] && [ "$short_test" -le "$tol_test" ]; then
      echo "delivered $have_train/$N_TRAIN train, $have_test/$N_TEST test"
      break
    fi
    if [ "$round" -gt "${#rounds[@]}" ]; then
      echo "topup: out of round prefixes" >&2; exit 1
    fi

    prefix="${rounds[round - 1]}"
    n_train="$(build_n "$short_train" "$yield")"
    n_test="$(build_n "$short_test" "$yield")"
    echo "--- round $round ($prefix): short $short_train train, $short_test test"
    echo "    building $n_train train, $n_test test at yield $yield"

    NAME="intent_build_$prefix" CPUS=8 MEM=96G TIME=03:00:00 \
      bash src/generalist/tools/run_py.sh -m src.generalist.assistant.pipeline.build \
        --config "$CONFIG" --cell "$CELL" --out "$OUT/topup-$prefix" \
        --n-train "$n_train" --n-test "$n_test" \
        --seed "$((SEED + round))" --id-prefix "$prefix" --domain "$DOMAIN"
    cp "$OUT/topup-$prefix/$prefix"*batch-*.json "$OUT/"

    pass_ask   "_$prefix" "$prefix*batch-*.json"
    pass_voice "_$prefix" "$prefix*batch-*.json"
    pass_judge "_$prefix" "$prefix*batch-*.json"
    run_accept

    now="$(accepted_field '.accepted // 0')"
    gained=$(( now - had ))
    yield="$(awk -v g="$gained" -v n="$((n_train + n_test))" \
                 'BEGIN { printf "%.4f", (n > 0 ? g / n : 0) }')"
    echo "--- round $round added $gained rows ($had -> $now), marginal yield $yield"
    # A round that converts almost nothing is the molecule pool saturating, not a
    # round that needs repeating: every new draw is a near-duplicate of something
    # already accepted, and the next round would buy the same nothing for the same
    # GPU-hour. Stop and let the shortfall be a decision rather than a loop.
    if [ "$gained" -lt $(( (short_train + short_test) / 10 )) ]; then
      echo "topup: round $round converted $gained rows against a shortfall of" \
           "$(( short_train + short_test )) — the pool is saturating, stopping" \
           "short rather than spending another round on it" >&2
      break
    fi
  done
fi

if should_run compose; then
  echo "=== compose ==="
  NAME=intent_compose CPUS=8 MEM=32G TIME=01:00:00 \
    bash src/generalist/tools/run_py.sh -m src.generalist.assistant.pipeline.compose \
      --accepted "$OUT/accepted" --out "$OUT/composed" --seed "$SEED" \
      --domain "$DOMAIN"
fi

echo
echo "done. Next, by hand and in this order:"
echo "  assistant.analysis.calibrate --mode sheet  --batches $OUT --asks $OUT/ask --voiced $OUT/voice --judged $OUT/judged --out $OUT/calibration"
echo "  (label $OUT/calibration/labels.jsonl, then --mode score)"
echo "  assistant.analysis.audit --mode sheet --composed $OUT/composed --out $OUT/audit"
echo "  (read $OUT/audit/sheet.txt, label $OUT/audit/labels.jsonl, then --mode score)"
