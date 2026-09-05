# CLAUDE.md — doze-llm

Read in this order at the start of every session: `CONTEXT.md` (history,
decisions, what is frozen vs tunable, next steps), `PREREG.md` (the frozen
experimental design), `README.md`. Then run `python -m pytest -q` and
`ls results/` and confirm the repo matches CONTEXT.md §6.

## Role

Builder and operator. The design is settled in PREREG.md. CONTEXT.md §7 says
exactly what is frozen and what is tunable. Do not change a frozen item
without stopping and flagging it; if unavoidable, record it in
`DEVIATIONS.md` with a reason. Prompt wording, token budgets, probe
statistics and implementation details are tunable.

## Workflow

- Work through CONTEXT.md §8 in order. For each step: tests first against
  what PREREG says, implement, run the full suite, then the smallest real
  check that exercises the new code: scripted backend → 4-bit smoke → bf16
  for any number that will be recorded.
- Report results as the header line + JSON + one or two raw completions.
- Keep CONTEXT.md §5 (calibration record) and §6 (repo state) current.
- When a result contradicts an expectation written in CONTEXT.md, say so
  plainly and stop for a decision. Do not adjust the design to fit.
- When the arms are implemented, ask for a review against PREREG by a fresh
  session before any Modal run.

## Conventions (CONTEXT.md §9)

- Run everything from the repo root with the venv active
  (`.venv/bin/python`). Python 3.14.
- Tests are model-free and must stay green; run `python -m pytest -q`
  before and after every change.
- Never put shell comments containing apostrophes on copy-paste lines (zsh
  opens a quote and hangs on `quote>`).
- Edit files in place; never create copies. A stale copy already cost one
  calibration cycle.
- Do not announce routine actions. Do flag anything touching a frozen item,
  any code/PREREG inconsistency, and any result that contradicts CONTEXT.md.
- Cheap over clever: 4-bit for smoke tests, bf16 for recorded numbers,
  cloud (Transformers+PEFT on Modal) for grids. Never mix MLX and PEFT
  numbers in one comparison.

## Modal runs

Always `modal run --detach`. Every launch gets a run tag; files live under
`/results/<tag>/` in the `doze-results` volume. Preempted GPU functions
resume from their last checkpoint; a preempted orchestrator re-attaches.
Re-attach a launch with `--run-tag <tag>`. All arms of a seed on one GPU
type (default L4).

## Lightning AI Studios

Studio `doze-llm` (teamspace general). Launch jobs with `deploy/vultr/runner.sh`
and `DOZE_STOP_WHEN_DONE=1`; never leave auto-sleep off without a running job
(idle L4 ≈ 1.3 credits/h). Credentials live in `.env` (LIGHTNING_*), never in the repo.

## Vultr VM path (from 2026-09-05)

`deploy/vultr/`: rsync the repo to /opt/doze, run `setup.sh` once, then
`launch_seed.sh <seed>` (tmux per GPU, resumable). Results under
/data/results/<tag>/; pull with rsync into results/final/<tag>/. Never store
API keys in the repo; the Vultr key lives outside it (`~/.config/vultr/api_key`).

## GitHub

This project uses only the **Rj2790** GitHub account (`gh auth switch -u Rj2790`
before any gh or git remote operation). Remote: https://github.com/Rj2790/doze-llm.

## Silent-failure watchlist

These would make arms non-comparable without crashing. Test for each:
compute budgets drifting outside ±5%; replay buffer or dream generator
touching a held-out prefix; off-by-one on when a night triggers; any dream
entering training without verification against the ground-truth solver;
MLX and PEFT numbers in one comparison.
