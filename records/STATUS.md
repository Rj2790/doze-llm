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

## Teardown when both queues show QUEUE_DONE (executed 2026-09-12, see TIMELINE)

    ./deploy/vultr/pull_results.sh
    curl -H "Authorization: Bearer $VULTR" -X DELETE https://api.vultr.com/v2/instances/8b8aa0a1-5679-46b8-8034-ef9a8d225867

## Pending decisions / blockers

- Grid done; VM destroyed 2026-09-12 11:02 IST. Modal: no running apps; Vultr:
  instance list empty; Lightning Studio was stopped after seed 1 (not
  re-verified via API today).
- Decision for the user (CONTEXT §8 item 16): stop and write up the null;
  8B dense confirmation; or the larger-model extension. The 4B data say no
  arm ever left chance on the probe, so the task difficulty, not the update
  schedule, is the binding constraint.
- Analysis command (re-runnable from this folder alone):
  `python -m eval.analyze --final --results records --seeds 0,1,2,3,4`.

## Known caveats for the analysis (keep with the data)

- **Seed 1 mixes GPU types.** Sleep and Baseline ran on a Lightning L4; Online,
  Sleep-NoDream and Awake ran on the Vultr A16. Same code, deterministic mode
  on both, but the untrained episode-0 evaluation differs slightly between
  GPU types (seed 1 on L4: probe 0.417 / held-out 0.478 / GSM8K 0.577; on
  A16: 0.400 / 0.463 / 0.583). Seeds 2–4 are entirely on the A16; seed 0 is
  entirely on Modal L4 (pilot code for Sleep/Online/Awake). Report per-seed
  comparisons within GPU type where it matters; note the seed-1 mix.
- **Pilot vs post-pilot code.** Seed 0's Sleep/Online/Awake used the one-step
  dreamer and non-deterministic CUDA; everything later uses the two-step
  dreamer (duplicate/out-of-spec rejection) and deterministic CUDA
  (DEVIATIONS.md 2026-09-04). Seed 0's Sleep therefore had a different dream
  stream from seeds 1–4.
- **Secondary metrics exist only for post-pilot runs** (all seeds ≥ 1 and
  seed 0's Sleep-NoDream/Baseline); mirror_bias becomes uninformative when
  held-out accuracy > 0.9 (few error steps); short_gap is only readable while
  short-mode unparsable is ~0.
- **Frozen arms:** Awake/Baseline checkpoints are identical by construction on
  one GPU; `frozen_eval_identical` in the analysis confirms it per run.

## Progress table

| seed | sleep | sleep_nodream | online | baseline | awake | matched |
|---|---|---|---|---|---|---|
| 0 | ✓ pilot (Modal L4) | ✓ (Modal L4) | ✓ pilot | ✓ (Modal L4) | ✓ pilot | ✓ |
| 1 | ✓ (Lightning L4) | ✓ (A16, 8.5 h) | ✓ (A16, 8.6 h) | ✓ (Lightning L4) | ✓ (A16, 17.4 h) | ✓ MATCHED |
| 2 | ✓ (A16) | ✓ (A16, 8.7 h) | ✓ (A16, 8.4 h) | ✓ (A16, 5.2 h) | ✓ (A16, 18.4 h) | ✓ MATCHED |
| 3 | ✓ (A16) | ✓ (A16, 8.3 h) | ✓ (A16, 8.7 h) | ✓ (A16, 5.2 h) | ✓ (A16, 17.6 h) | ✓ MATCHED |
| 4 | ✓ (A16) | ✓ (A16, 8.4 h) | ✓ (A16, 9.1 h) | ✓ (A16, 5.2 h) | ✓ (A16, 17.7 h) | ✓ MATCHED |

Seed 1 Sleep-NoDream (A16, 8.52 h): probe 0.33–0.47 (chance), held-out
0.463 → 0.886 (peak 0.930 at ep 500), GSM8K 0.583 → 0.553; 600 steps,
58,782 training tokens, 58,882 generated. Seed 1 Awake (A16, 17.4 h):
101,422 generated tokens vs Sleep 101,136 (+0.3%, matched); frozen eval
0.350 / 0.453 / 0.587 at all 13 checkpoints.

Projected completion of the whole grid: 13 Sep 2026 ~06:30 IST.

Seed 1 complete (all five arms), `check_seed` MATCHED (awake tokens +0.28%,
online steps −1.83% / train tokens −2.24%, nodream train +0.61% vs Sleep).
Final held-out: Sleep 0.811, NoDream 0.886, Online 0.900, Awake 0.453,
Baseline 0.478. GSM8K delta: Sleep −0.080, NoDream −0.030, Online +0.177.
Probe: all at chance, all censored. See `seed1/report.md`.

Projected grid completion revised to 12 Sep ~12:30 IST (Baselines 5.2 h, not 7).

**GRID COMPLETE 2026-09-12 10:35 IST.** 25/25 arms; both queues QUEUE_DONE;
every seed MATCHED. Final analysis: `analysis_final.md` / `.json`
(H1–H4 all not supported; see `../CONTEXT.md` §5k for the reading).
Vultr GPU-hours on the grid: 181.8 (18 arms). Pending charges at teardown:
$89.50 of the $300 credit. Trained-arm adapter states (Sleep / NoDream /
Online, seeds 1–4; `<arm>_seedN.json.state/` with `backend.pt`, `buffer.json`,
`sleep.json`) archived off-repo at
`~/doze-archive/vultr-final/trained_arm_states_seeds1-4.tar` (~4.8 GB) before
the VM was destroyed. Frozen-arm state dumps (8 GB full-model each) and the
validation run were not kept.

Vultr instance 8b8aa0a1 destroyed 2026-09-12 11:02 IST (API DELETE 204; instance list empty).

Local archive complete (2026-09-12 12:30 IST): 13 trained adapters exported
to PEFT format under `~/doze-archive/models/adapters/`; raw states, the
Modal volume and the Lightning seed-1 tar alongside; base model downloading
into `models/base/`. Seed 0 pilot Sleep/Online adapters never existed on disk
(pilot code saved no state). Lightning Studio started on CPU for the fetch
and stopped again (status verified Stopped). Layout: `README.md` in this folder.
Deep exploratory analysis: `analysis_deep.md` + `analysis_deep_notes.md`.

Last updated: 2026-09-12 12:30 IST
