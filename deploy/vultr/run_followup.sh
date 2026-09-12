#!/usr/bin/env bash
# Run the post-hoc follow-up (eval/followup.py) on a Vultr GPU VM.
# Usage on the VM (after setup.sh, repo at /opt/doze, adapters at /data/adapters):
#   deploy/vultr/run_followup.sh <gpu> <only: A|B|C|A,B,C>
# Resumable: every summary file is re-read and finished keys are skipped.
set -euo pipefail
GPU=$1; ONLY=${2:-A,B,C}
DOZE_ROOT=${DOZE_ROOT:-/opt/doze}; export DOZE_RESULTS=${DOZE_RESULTS:-/data/results}; export HF_HOME=${HF_HOME:-/data/hf-cache}
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8
OUT=$DOZE_RESULTS/followup
mkdir -p "$OUT"
cd "$DOZE_ROOT"
PY=$DOZE_ROOT/.venv/bin/python
echo "$(date -u +%FT%TZ) gpu$GPU start followup only=$ONLY" >> "$OUT/runner_gpu$GPU.log"
"$PY" -m eval.followup --out "$OUT" --adapters-root /data/adapters --only "$ONLY" --seeds 0,1,2,3,4 \
  >> "$OUT/followup_gpu$GPU.out" 2>&1 && STATUS=0 || STATUS=$?
echo "$(date -u +%FT%TZ) gpu$GPU done followup only=$ONLY exit=$STATUS" >> "$OUT/runner_gpu$GPU.log"
