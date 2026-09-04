#!/usr/bin/env bash
# Launch all five arms of one seed across the VM's GPUs (detached, tmux). Usage: ./deploy/vultr/launch_seed.sh <seed> [tag]
set -euo pipefail
SEED=$1; TAG=${2:-seed$1}
cd /opt/doze
NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
QUEUES=$(.venv/bin/python -c "
from deploy.vultr.gpu_queue import plan_queues
for q in plan_queues(['sleep','sleep_nodream','online','baseline','awake'], $NGPU): print(' '.join(q))")
i=0
while IFS= read -r q; do
  tmux new-session -d -s "doze_${TAG}_gpu$i" "./deploy/vultr/runner.sh $i $SEED $TAG $q"
  echo "gpu$i: $q"; i=$((i+1))
done <<< "$QUEUES"
