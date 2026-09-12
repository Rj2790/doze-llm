# records/ — durable run artefacts (committed)

`results/` is the git-ignored working area; everything worth keeping is copied
here. Run files are the harness's `RunResult.save` JSON (config, per-episode
trajectories with text, per-checkpoint metrics incl. the post-pilot secondary
metrics where available, nights with per-dream records, ledger, matching
verdict). Load with `arms.harness.load_run`. Ledgers load with
`eval.compute_ledger.Ledger.load`. `eval/analyze.py --results <dir>` works on
any directory with `runs/` and `ledgers/` subfolders (e.g. `records/seed0`).

| path | what |
|---|---|
| `seed0/runs/`, `seed0/ledgers/` | seed 0, all five arms. Sleep/Online/Awake = pilot (Modal L4, non-deterministic CUDA, one-step dreamer); Sleep-NoDream/Baseline = post-pilot code (deterministic, secondary metrics). `check_seed` verdict MATCHED (in each run file under `matched`). |
| `seed0/report.md`, `summary.json`, `grid_seed0.json` | analysis output and the Modal grid summary for the NoDream+Baseline launch |
| `seed1/` … `seed4/` | `runs/` (five arms each, complete), `ledgers/`, `logs/`, `*.timing.json` (GPU, total seconds, per-checkpoint seconds), `report.md` (per-seed `check_seed` + summary). Seed 1: Sleep and Baseline on Lightning L4, others Vultr A16; seeds 2–4 all Vultr A16, deterministic CUDA. |
| `analysis_final.md`, `analysis_final.json` | the preregistered five-seed analysis (PREREG §7): log-rank (censored), paired sign-flip H4, per-arm summaries, exploratory schedule comparison. `analysis_preliminary.*` is the same report run before Awake 2/3/4 and Baseline 4 finished. |
| `STATUS.md`, `TIMELINE.md` | operational record: instances, queues, pace, costs, caveats, progress table; dated event log |
| `calibration/` | untrained-model calibration runs 2–6 (MLX), backend agreement MLX vs HF on 40 items, HF pre-flight (validation items 1–4, agreement set, dream stats) |
| `retro/retro_frozen_seed0.json` | secondary metrics + GSM8K parse rate retro-computed for the frozen arms (Baseline = base model, Awake) |
| `infra/` | A100 path check, deterministic on/off timing pair (L4, Modal), Lightning L4 validation timing |

Narrative, decisions and numbers: `../CONTEXT.md` §5–§5k (§5k = final result). Timeline of runs,
failures and costs: `TIMELINE.md`.
