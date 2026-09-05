#!/usr/bin/env bash
# Run one queue of arms sequentially on one GPU, resumable. Usage:
#   ./deploy/vultr/runner.sh <gpu_index> <seed> <tag> arm1 arm2 ...
# 'awake' derives its budget from the sleep ledger of the same tag/seed.
# Paths: DOZE_ROOT (repo), DOZE_RESULTS (results root), HF_HOME (model cache); defaults are the Vultr layout.
set -uo pipefail
GPU=$1; SEED=$2; TAG=$3; shift 3
DOZE_ROOT=${DOZE_ROOT:-/opt/doze}; export DOZE_RESULTS=${DOZE_RESULTS:-/data/results}; export HF_HOME=${HF_HOME:-/data/hf-cache}
PY=${DOZE_PYTHON:-$DOZE_ROOT/.venv/bin/python}; [ -x "$PY" ] || PY=python
BATCH=${DOZE_BATCH:-8}
cd "$DOZE_ROOT"
export CUDA_VISIBLE_DEVICES=$GPU TOKENIZERS_PARALLELISM=false
mkdir -p "$DOZE_RESULTS/$TAG/logs"
for ARM in "$@"; do
  EXTRA=""; [ "$ARM" = "awake" ] && EXTRA="--awake-from-sleep"
  echo "$(date -u +%FT%TZ) gpu$GPU start $ARM seed$SEED" >> "$DOZE_RESULTS/$TAG/logs/runner_gpu$GPU.log"
  "$PY" deploy/vultr/run_job.py --arm $ARM --seed $SEED --run-tag $TAG --results "$DOZE_RESULTS" --batch-size $BATCH $EXTRA \
      >> "$DOZE_RESULTS/$TAG/logs/${ARM}_seed${SEED}.log" 2>&1
  echo "$(date -u +%FT%TZ) gpu$GPU done  $ARM seed$SEED exit=$?" >> "$DOZE_RESULTS/$TAG/logs/runner_gpu$GPU.log"
done
echo "$(date -u +%FT%TZ) gpu$GPU QUEUE_DONE" >> "$DOZE_RESULTS/$TAG/logs/runner_gpu$GPU.log"
# Lightning Studios: stop the machine when the queue is done so an idle GPU never bills (2026-09-05: ~14 credits lost idling)
if [ "${DOZE_STOP_WHEN_DONE:-0}" = "1" ]; then
  "$PY" -c "from lightning_sdk import Studio; Studio().stop()" >> "$DOZE_RESULTS/$TAG/logs/runner_gpu$GPU.log" 2>&1 || true
fi
