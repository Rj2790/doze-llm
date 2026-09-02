# sleep-cycle

Does phased offline consolidation ("sleep") beat continuous updates and
inference-time reasoning at matched compute? See `PREREG.md`.

## Status

- [x] Preregistration draft
- [x] Number Reduction task (generator, prefix-disjoint splits, prompts, scoring)
- [x] Shortcut detector (probe accuracy + token collapse) with scripted-solver tests
- [x] Backend interface + scripted backend + MLX backend (generation only)
- [x] `calibrate.py` — untrained-model difficulty check (stratified probe, `--mode`, `--number-digits`)
- [x] Control benchmark loader (300 fixed GSM8K test items)
- [x] Compute ledger with ±5% matching check
- [x] Arms: baseline, awake, online, sleep, sleep-nodream (`run_arm.py`), MLX LoRA backend
- [x] Fresh-session review of arms vs PREREG (`REVIEW.md`); amendments applied; PREREG v1.0 frozen
- [ ] HF/PEFT backend + Modal app (written, not yet run) — next: path check
- [ ] String grammar task
- [ ] Analysis script

## Setup (Mac)

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python -m pytest -q

## Calibrate (the first thing to run with a real model)

    python calibrate.py --backend scripted                                   # dry run
    python calibrate.py --backend mlx --model mlx-community/Qwen3-4B-4bit --n 60 --n-probe 60
    python calibrate.py --backend mlx --model mlx-community/Qwen3-4B-bf16 --n 200 --n-probe 60

Writes `results/calibration.json`. See the docstring in `calibrate.py` for
how to read the verdict.

## Run an arm

    python run_arm.py --arm baseline --backend scripted --n-episodes 100 --k 50 --n-probe 30 --n-heldout-eval 20
    python run_arm.py --arm sleep --backend mlx --model mlx-community/Qwen3-4B-4bit --n-episodes 4 --k 2 --n-probe 6 --n-heldout-eval 2

Read `CONTEXT.md` first; `CLAUDE.md` has the working conventions.
