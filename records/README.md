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
| `seed1/sleep_seed1.partial.json` | seed 1 Sleep (Lightning Studio L4, two-step dreamer), partial snapshot; replaced by the complete file when the run ends |
| `calibration/` | untrained-model calibration runs 2–6 (MLX), backend agreement MLX vs HF on 40 items, HF pre-flight (validation items 1–4, agreement set, dream stats) |
| `retro/retro_frozen_seed0.json` | secondary metrics + GSM8K parse rate retro-computed for the frozen arms (Baseline = base model, Awake) |
| `infra/` | A100 path check, deterministic on/off timing pair (L4, Modal), Lightning L4 validation timing |

Narrative, decisions and numbers: `../CONTEXT.md` §5–§5h. Timeline of runs,
failures and costs: `TIMELINE.md`.
