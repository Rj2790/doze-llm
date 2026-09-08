#!/usr/bin/env bash
# VM-side orchestrator for many (seed, arm) items across all visible GPUs.
# Survives SSH disconnects (nohup); every job is resumable; each seed's files
# live under $DOZE_RESULTS/seed<N>/. Usage:
#   ./deploy/vultr/run_all.sh "1:online 1:sleep_nodream 1:awake 2:sleep 2:sleep_nodream 2:online 2:baseline 2:awake ..."
# Env: DOZE_ROOT, DOZE_RESULTS, HF_HOME, DOZE_BATCH, DOZE_PYTHON (see runner.sh).
set -euo pipefail
ITEMS=$1
DOZE_ROOT=${DOZE_ROOT:-/opt/doze}; export DOZE_RESULTS=${DOZE_RESULTS:-/data/results}
PY=${DOZE_PYTHON:-$DOZE_ROOT/.venv/bin/python}; [ -x "$PY" ] || PY=python
cd "$DOZE_ROOT"
NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
mkdir -p "$DOZE_RESULTS/logs"
QUEUES=$("$PY" -c "
import sys
from deploy.vultr.gpu_queue import plan_multi
items = [(int(s), a) for s, a in (x.split(':') for x in sys.argv[1].split())]
for q in plan_multi(items, int(sys.argv[2])): print(' '.join(f'{s}:{a}' for s, a in q))" "$ITEMS" "$NGPU")
i=0
while IFS= read -r q; do
  nohup ./deploy/vultr/runner_multi.sh $i $q > "$DOZE_RESULTS/logs/nohup_gpu$i.out" 2>&1 &
  echo "gpu$i (pid $!): $q"; i=$((i+1))
done <<< "$QUEUES"
