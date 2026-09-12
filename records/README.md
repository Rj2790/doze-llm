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
| `analysis_deep.md`, `analysis_deep.json`, `analysis_deep_notes.md`, `figures/` | exploratory deep analysis of the completed grid (`python -m eval.deep_analysis`): pooled probe tests, day-time learning curves, error anatomy, per-item held-out recheck, GSM8K dynamics and night correlations, night-by-night dream statistics, buffer composition, stability, Awake rounds, compute, power; `_notes.md` is the human reading of it |
| `seedN/state/` | final replay buffers (`<arm>_seedN.buffer.json`) and night counters (`.sleep.json`) recovered from the resume-state directories of Sleep / Sleep-NoDream (seed 0 NoDream from Modal; seed 1 Sleep from Lightning; the rest from Vultr) |
| `analysis_final.md`, `analysis_final.json` | the preregistered five-seed analysis (PREREG §7): log-rank (censored), paired sign-flip H4, per-arm summaries, exploratory schedule comparison. `analysis_preliminary.*` is the same report run before Awake 2/3/4 and Baseline 4 finished. |
| `STATUS.md`, `TIMELINE.md` | operational record: instances, queues, pace, costs, caveats, progress table; dated event log |
| `calibration/` | untrained-model calibration runs 2–6 (MLX), backend agreement MLX vs HF on 40 items, HF pre-flight (validation items 1–4, agreement set, dream stats) |
| `retro/retro_frozen_seed0.json` | secondary metrics + GSM8K parse rate retro-computed for the frozen arms (Baseline = base model, Awake) |
| `infra/` | A100 path check, deterministic on/off timing pair (L4, Modal), Lightning L4 validation timing |

Narrative, decisions and numbers: `../CONTEXT.md` §5–§5k (§5k = final result). Timeline of runs,
failures and costs: `TIMELINE.md`.

## Off-repo archive (local disk, `~/doze-archive/`)

Large artefacts that do not belong in git. Laid out on 2026-09-12 after the
grid finished and all remote compute was torn down.

| path | what |
|---|---|
| `models/adapters/<arm>_seedN/` | **13 trained LoRA adapters in standard PEFT format** (`adapter_model.safetensors` 126 MB fp32 + `adapter_config.json` + `export_info.json`): Sleep, Sleep-NoDream and Online for seeds 1–4, plus Sleep-NoDream seed 0. Load with `PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B"), path)`. Exported by `deploy/export_adapter.py`; smoke-tested with `deploy/verify_adapter.py` (sleep_seed3: 6/6 held-out vs base 2/6). |
| `models/seedN/<arm>_seedN/` | the raw harness resume states those were exported from (`backend.pt` = LoRA + Adam + RNG, `buffer.json`, `sleep.json` / `online.json`) |
| `models/base/Qwen3-4B/` | the unmodified base model (HF safetensors). This *is* the Baseline and Awake "model": those arms never update weights. |
| `vultr-final/trained_arm_states_seeds1-4.tar` | verbatim tar of the 12 Vultr state dirs (4.37 GB, checksum-verified against the VM before it was destroyed) |
| `lightning-seed1/seed1_sleep_state_and_logs.tar` | seed 1 Sleep state + Lightning logs (md5 d6568617…) |
| `modal-doze-results/` | the whole Modal `doze-results` volume (16 GB): seed 0 runs/ledgers, seed 0 NoDream/Baseline/Awake states, the det-on/det-off timing pair, preflight and retro files |

**Not recoverable:** the seed 0 *pilot* Sleep and Online adapters. The pilot
code did not save resume state and the Modal volume holds only their run and
ledger files. The seed 0 Baseline/Awake state dumps (7.5 GB full-model copies
identical to the base weights) were left in `modal-doze-results/` and not
duplicated into `models/`.
