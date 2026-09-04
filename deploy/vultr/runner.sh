#!/usr/bin/env bash
# Run one queue of arms sequentially on one GPU, resumable. Usage:
#   ./deploy/vultr/runner.sh <gpu_index> <seed> <tag> arm1 arm2 ...
# 'awake' derives its budget from the sleep ledger of the same tag/seed.
set -uo pipefail
GPU=$1; SEED=$2; TAG=$3; shift 3
cd /opt/doze
export CUDA_VISIBLE_DEVICES=$GPU HF_HOME=/data/hf-cache DOZE_RESULTS=/data/results TOKENIZERS_PARALLELISM=false
mkdir -p /data/results/$TAG/logs
for ARM in "$@"; do
  EXTRA=""; [ "$ARM" = "awake" ] && EXTRA="--awake-from-sleep"
  echo "$(date -u +%FT%TZ) gpu$GPU start $ARM seed$SEED" >> /data/results/$TAG/logs/runner_gpu$GPU.log
  .venv/bin/python deploy/vultr/run_job.py --arm $ARM --seed $SEED --run-tag $TAG --results /data/results $EXTRA \
      >> /data/results/$TAG/logs/${ARM}_seed${SEED}.log 2>&1
  echo "$(date -u +%FT%TZ) gpu$GPU done  $ARM seed$SEED exit=$?" >> /data/results/$TAG/logs/runner_gpu$GPU.log
done
echo "$(date -u +%FT%TZ) gpu$GPU QUEUE_DONE" >> /data/results/$TAG/logs/runner_gpu$GPU.log
