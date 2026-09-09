#!/usr/bin/env bash
# Pull everything worth keeping from the VM into records/ (partial runs included;
# adapter/optimizer state dirs excluded — too large for git and reproducible).
# Usage: ./deploy/vultr/pull_results.sh [host]   (default host from records/STATUS.md)
set -euo pipefail
HOST=${1:-$(grep -m1 -oE 'IP: [0-9.]+' records/STATUS.md | cut -d' ' -f2)}
R="rsync -az -e ssh"
for s in 1 2 3 4; do
  mkdir -p records/seed$s/runs records/seed$s/ledgers records/seed$s/logs
  $R --exclude '*.state' root@$HOST:/data/results/seed$s/runs/ records/seed$s/runs/ 2>/dev/null || true
  $R root@$HOST:/data/results/seed$s/ledgers/ records/seed$s/ledgers/ 2>/dev/null || true
  $R root@$HOST:/data/results/seed$s/logs/ records/seed$s/logs/ 2>/dev/null || true
  # timing files live next to the run files
  for f in records/seed$s/runs/*.timing.json; do [ -f "$f" ] && mv -f "$f" records/seed$s/ 2>/dev/null || true; done
done
mkdir -p records/infra/vm-logs
$R root@$HOST:/data/results/logs/ records/infra/vm-logs/ 2>/dev/null || true
$R root@$HOST:/data/results/val/ records/infra/vm-logs/val/ --exclude '*.state' 2>/dev/null || true
echo "pulled from $HOST at $(date -u +%FT%TZ)"
