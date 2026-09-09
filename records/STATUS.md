# STATUS — where the experiment stands and how to resume it without this session

Updated: see the "Last updated" line at the bottom (kept current by the operator).
Read `README.md`, `TIMELINE.md` and `../CONTEXT.md` §5–§5i for the narrative.

## Compute in use

- **Vultr instance** `doze-llm-a16-sgp`, id `8b8aa0a1-5679-46b8-8034-ef9a8d225867`,
  plan `vcg-a16-12c-128g-32vram` (2 × NVIDIA A16-16Q), region sgp, Ubuntu 22.04.5,
  **IP: 207.148.68.114**, root via SSH key `rac-macbook` (~/.ssh/id_ed25519).
  $0.942/h, billed while it exists (stopped instances still bill). Created
  2026-09-08 07:39 UTC; grid launched 07:59 UTC.
- Repo on the VM: `/opt/doze` (venv `.venv`, torch 2.5.1+cu121). Results:
  `/data/results/seed<N>/{runs,ledgers,logs}`, model cache `/data/hf-cache`.
- Vultr API key: `.env` (`VULTR=`), git-ignored. The key's IP allow-list must
  include the laptop's current IP (Account → API → Access Control).
- Lightning Studio `doze-llm` (teamspace general): stopped; ~7 credits left.
- Modal: wallet ~$1; not in use.

## What is running (VM-side, nohup, resumable; survives laptop/session loss)

Launched with `deploy/vultr/run_all.sh` → two `runner_multi.sh` queues:
- gpu0: 2:sleep ✓ 4:sleep ✓ 1:sleep_nodream 2:online 3:online 4:online 2:awake 4:awake
- gpu1: 3:sleep ✓ 1:awake 1:online 2:sleep_nodream 3:sleep_nodream 4:sleep_nodream 2:baseline 3:baseline 4:baseline 3:awake
Per-item logs: `/data/results/seed<N>/logs/<arm>_seed<N>.log`; queue logs
`/data/results/logs/runner_gpu{0,1}.log` (lines: start / done exit=… / QUEUE_DONE).
Measured pace on A16 (deterministic): LoRA arms ~8.5–9 h (13.5 s/episode,
27 min/checkpoint), Awake ~18.5 h (71 min/checkpoint), Baseline ~7 h.

## How to check progress (from any machine with the SSH key)

    ssh root@207.148.68.114 'cat /data/results/logs/runner_gpu*.log; for f in /data/results/seed*/logs/*_seed*.log; do grep -h "\] ep=" $f | tail -1; done; nvidia-smi'

## How to pull results into records/

    ./deploy/vultr/pull_results.sh            # runs, ledgers, timings, logs (partial files included)
    .venv/bin/python -m eval.analyze --results records/seed<N> --seeds <N>   # per seed once its ledgers exist

`eval/analyze.check_seed(records/seed<N>, seed=<N>)` writes the matching
verdict into the run files; requires the seed's `sleep` ledger.

## If the VM dies or must be recreated

1. Create the same plan (any region with stock; API or console, Ubuntu 22.04).
2. `rsync -az --exclude .venv --exclude results --exclude .git ./ root@<ip>:/opt/doze/`
   then `ssh root@<ip> bash /opt/doze/deploy/vultr/setup.sh` (driver preinstalled on Vultr GPU images).
3. Restore the completed/partial files you need under `/data/results/seed<N>/`
   from `records/seed<N>/` (runs + ledgers; a partial run file + its `.state`
   dir are needed to resume mid-run — state dirs are NOT in git, so a lost VM
   means restarting the interrupted arms from scratch).
4. Relaunch the remaining items with `run_all.sh "<seed>:<arm> ..."`; Awake items
   need the seed's Sleep ledger present.

## Teardown when both queues show QUEUE_DONE

    ./deploy/vultr/pull_results.sh
    curl -H "Authorization: Bearer $VULTR" -X DELETE https://api.vultr.com/v2/instances/8b8aa0a1-5679-46b8-8034-ef9a8d225867

## Pending decisions / blockers

- A second instance would halve the remaining wall-clock at the same total
  cost; blocked by the account's monthly fee cap (one $688/mo plan). Limit
  increase requested by the user; if granted, split the not-yet-started items
  across the new VM's GPUs.
- After the grid: run `eval/analyze.py` over all seeds, write CONTEXT §5j,
  then the analysis plan of PREREG §7 (log-rank on episode-to-criterion —
  all censored so far — and the paired forgetting comparison).

## Progress table

| seed | sleep | sleep_nodream | online | baseline | awake | matched |
|---|---|---|---|---|---|---|
| 0 | ✓ pilot (Modal) | ✓ | ✓ pilot | ✓ | ✓ pilot | ✓ |
| 1 | ✓ (Lightning) | running gpu0 | queued gpu1 | ✓ (Lightning) | running gpu1 (ep 550) | — |
| 2 | ✓ | queued | queued | queued | queued | — |
| 3 | ✓ | queued | queued | queued | queued | — |
| 4 | ✓ | queued | queued | queued | queued | — |

Last updated: 2026-09-09 08:30 UTC (14:00 IST)
