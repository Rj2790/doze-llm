#!/usr/bin/env bash
# Launch all five arms of one seed across the visible GPUs, one detached runner per GPU.
# Usage: ./deploy/vultr/launch_seed.sh <seed> [tag]   (env: DOZE_ROOT, DOZE_RESULTS, HF_HOME, DOZE_BATCH, DOZE_PYTHON)
set -euo pipefail
SEED=$1; TAG=${2:-seed$1}
DOZE_ROOT=${DOZE_ROOT:-/opt/doze}; cd "$DOZE_ROOT"
PY=${DOZE_PYTHON:-$DOZE_ROOT/.venv/bin/python}; [ -x "$PY" ] || PY=python
NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
QUEUES=$("$PY" -c "
from deploy.vultr.gpu_queue import plan_queues
for q in plan_queues(['sleep','sleep_nodream','online','baseline','awake'], $NGPU): print(' '.join(q))")
mkdir -p "${DOZE_RESULTS:-/data/results}/$TAG/logs"
i=0
while IFS= read -r q; do
  nohup ./deploy/vultr/runner.sh $i $SEED $TAG $q > "${DOZE_RESULTS:-/data/results}/$TAG/logs/nohup_gpu$i.out" 2>&1 &
  echo "gpu$i: $q (pid $!)"; i=$((i+1))
done <<< "$QUEUES"
