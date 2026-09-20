#!/usr/bin/env bash
# =============================================================================
# intent_pipeline.sh — build the §9.4 assistant set, end to end.
# =============================================================================
# Six stages in the order §9.4 fixes, because a change to an earlier one
# invalidates everything written after it:
#
#   build   declare the intents and render their replies          (CPU)
#   ask     write the person's turn from the situation and the ask (GPU)
#   voice   re-voice the rendered reply under the style brief      (GPU)
#   judge   responsive / preserved / added, greedy                 (GPU)
#   accept  the three checks, the held-out filter, dedup, ceilings (CPU)
#   compose few-shot demonstrations drawn from accepted rows       (CPU)
#
# Each stage is a separate blocking sbatch, so a stage that fails stops the run
# with its own log rather than poisoning the next one. Re-running with the same
# --out skips nothing: stages are cheap to redo and a half-written stage is the
# one thing worth never trusting.
#
#   src/generalist/tools/intent_pipeline.sh --out src/generalist/results/assistant/v5 \
#       --n-train 11000 --n-test 900
#   src/generalist/tools/intent_pipeline.sh --out .../v5 --from voice
#
# `--from` restarts at a named stage. The writer stages need a card that fits a
# 31B model in bf16, which is why GPU_CONSTRAINT is set here and not left to the
# launcher's default — that default admits the 40G A100s, and a 31B writer does
# not fit on one.
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

OUT=""
CONFIG="src/generalist/configs/probes/008_molecule_generalist_instruct.jsonc"
CELL="molecule_generalist_instruct_graph_s0"
N_TRAIN=11000
N_TEST=900
FROM="build"
SEED=0
MODEL="/shared/workspace/povejmo/huggingface_cache/hub/models--google--gemma-4-31B-it/snapshots/b9ea41a2887d8607f594846523f94c6cc75ac8a4"

while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --cell) CELL="$2"; shift 2 ;;
    --n-train) N_TRAIN="$2"; shift 2 ;;
    --n-test) N_TEST="$2"; shift 2 ;;
    --from) FROM="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
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

STAGES=(build ask voice judge accept compose)
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
  NAME=intent_build CPUS=8 MEM=96G TIME=03:00:00 \
    bash src/generalist/tools/run_py.sh -m src.generalist.tools.intent_build \
      --config "$CONFIG" --cell "$CELL" --out "$OUT" \
      --n-train "$N_TRAIN" --n-test "$N_TEST" --seed "$SEED"
fi

if should_run ask; then
  echo "=== ask ==="
  env NAME=intent_ask "${WRITER_ENV[@]}" \
    bash src/generalist/tools/run_py.sh -m src.generalist.tools.intent_write \
      --model "$MODEL" --batches "$OUT" --out "$OUT/ask" --pass ask \
      --writer gemma-4-31B-it
fi

if should_run voice; then
  echo "=== voice ==="
  env NAME=intent_voice "${WRITER_ENV[@]}" \
    bash src/generalist/tools/run_py.sh -m src.generalist.tools.intent_write \
      --model "$MODEL" --batches "$OUT" --out "$OUT/voice" --pass voice \
      --asks "$OUT/ask" --writer gemma-4-31B-it
fi

if should_run judge; then
  echo "=== judge ==="
  env NAME=intent_judge "${WRITER_ENV[@]}" \
    bash src/generalist/tools/run_py.sh -m src.generalist.tools.intent_judge \
      --model "$MODEL" --batches "$OUT" --asks "$OUT/ask" \
      --voiced "$OUT/voice" --out "$OUT/judged"
fi

if should_run accept; then
  echo "=== accept ==="
  # --judged is on because the judge can only remove rows here, so its errors
  # cost yield and cannot cost correctness. On the final build that price was
  # measured by reading all 104 of its refusals: precision 0.231, so roughly 80
  # correct rows in 14,900. Two of the three misreadings behind that are fixed in
  # intent_judge.JUDGE_SYSTEM and its prompt; the number is stale the moment they
  # take effect, so read the refusals again rather than quoting 0.231 forward.
  # --dedup-max takes intent_accept.DEDUP_MAX, which is the settled 0.85.
  NAME=intent_accept CPUS=8 MEM=32G TIME=01:00:00 \
    bash src/generalist/tools/run_py.sh -m src.generalist.tools.intent_accept \
      --batches "$OUT" --asks "$OUT/ask" --voiced "$OUT/voice" \
      --judged "$OUT/judged" --out "$OUT/accepted" --seed "$SEED"
fi

if should_run compose; then
  echo "=== compose ==="
  NAME=intent_compose CPUS=8 MEM=32G TIME=01:00:00 \
    bash src/generalist/tools/run_py.sh -m src.generalist.tools.assistant_compose \
      --accepted "$OUT/accepted" --out "$OUT/composed" --seed "$SEED"
fi

echo
echo "done. Next, by hand and in this order:"
echo "  intent_calibrate --mode sheet  --batches $OUT --asks $OUT/ask --voiced $OUT/voice --judged $OUT/judged --out $OUT/calibration"
echo "  (label $OUT/calibration/labels.jsonl, then --mode score)"
echo "  intent_audit --mode sheet --composed $OUT/composed --out $OUT/audit"
echo "  (read $OUT/audit/sheet.txt, label $OUT/audit/labels.jsonl, then --mode score)"
