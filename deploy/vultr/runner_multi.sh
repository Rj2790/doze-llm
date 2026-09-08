#!/usr/bin/env bash
# Sequential queue of seed:arm items on one GPU. Usage: runner_multi.sh <gpu> 1:online 1:awake 2:sleep ...
set -uo pipefail
GPU=$1; shift
DOZE_ROOT=${DOZE_ROOT:-/opt/doze}; export DOZE_RESULTS=${DOZE_RESULTS:-/data/results}; export HF_HOME=${HF_HOME:-/data/hf-cache}
PY=${DOZE_PYTHON:-$DOZE_ROOT/.venv/bin/python}; [ -x "$PY" ] || PY=python
BATCH=${DOZE_BATCH:-8}
cd "$DOZE_ROOT"
export CUDA_VISIBLE_DEVICES=$GPU TOKENIZERS_PARALLELISM=false
LOG="$DOZE_RESULTS/logs/runner_gpu$GPU.log"
for ITEM in "$@"; do
  SEED=${ITEM%%:*}; ARM=${ITEM##*:}; TAG="seed$SEED"
  mkdir -p "$DOZE_RESULTS/$TAG/logs"
  EXTRA=""; [ "$ARM" = "awake" ] && EXTRA="--awake-from-sleep"
  echo "$(date -u +%FT%TZ) gpu$GPU start $ARM seed$SEED" >> "$LOG"
  "$PY" deploy/vultr/run_job.py --arm $ARM --seed $SEED --run-tag $TAG --results "$DOZE_RESULTS" --batch-size $BATCH $EXTRA \
      >> "$DOZE_RESULTS/$TAG/logs/${ARM}_seed${SEED}.log" 2>&1
  echo "$(date -u +%FT%TZ) gpu$GPU done  $ARM seed$SEED exit=$?" >> "$LOG"
done
echo "$(date -u +%FT%TZ) gpu$GPU QUEUE_DONE" >> "$LOG"
