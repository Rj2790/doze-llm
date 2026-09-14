# doze-llm

**A preregistered test of a "sleep cycle" for a small language model, and what it
taught us about measuring insight and forgetting.**

We asked whether giving Qwen3-4B a sleep-like training schedule (days of frozen
practice, nights of replay plus verified self-generated "dreams" and a burst of
LoRA steps) would help it discover a hidden shortcut in a digit puzzle sooner
than continuous updating or inference-time reasoning, at matched compute. Five
seeds, five regimes, 25 runs, everything frozen in advance.

The preregistered answer is a null on all four hypotheses. The more useful
findings came from taking the null apart:

1. **Verified self-distillation teaches a fixed procedure fast and generally.**
   Training on the model's own correct, verified solutions took held-out accuracy
   from 0.42 to 0.86–0.99 in 200–450 steps, equally under continuous or batched
   updates, and it generalised to strings without the practice structure
   (0.84–0.89 vs 0.37 untrained). Self-generated variants ("dreams") and
   inference-time self-critique added nothing.
2. **It also entrenches the procedure.** On a new puzzle whose rule alternates
   every step, all models score at chance and the trained ones are *worse* than
   the untrained model on the very first step.
3. **The insight probe could not be passed at this scale.** Answer-only masked
   items stayed at chance even for a model told the rule in the prompt or
   trained directly on the probe. The study could not have observed insight.
4. **The "forgetting" result was a prompt artefact.** The GSM8K control ran
   under a system prompt that forbids explanation (base 0.59). Under a normal
   math prompt the base model scores 0.91 and every trained adapter lands within
   ±0.02 of it.

Points 3 and 4 are the reason to read this repository: they are easy mistakes
to make and the evidence for each is in `records/`.

## Read this first

| document | what it is |
|---|---|
| [`REPORT.md`](REPORT.md) | the write-up: design, verdicts, deep analysis, the hypothesis-blind review, follow-up experiments |
| [`PREREG.md`](PREREG.md) | the preregistration, frozen at v1.0 before any run ([public gist](https://gist.github.com/Rj2790/5bfb6765d5491802c3e139d65afe3dc8)) |
| [`DEVIATIONS.md`](DEVIATIONS.md) | every change after the freeze, with reasons |
| [`CONTEXT.md`](CONTEXT.md) | the project log: decisions, calibration, every run, incidents, and the evolving reading (§5k–§5n) |
| [`records/`](records/README.md) | every run file, ledger, log, replay buffer, analysis and figure; durable and re-runnable |
| [`records/analysis_outsider.md`](records/analysis_outsider.md) | the hypothesis-blind reading, an independent second reading, and their reconciliation |

## The experiment in one paragraph

**Task.** Number Reduction, length 12 over {1,4,9}: fold left to right with a
fixed rule (same → same; different → the third digit), 11 steps, the last
result is the answer. Every training and evaluation string secretly satisfies
r11 = r6, so the answer is available five steps early. **Regimes.** Baseline
(frozen), Awake (frozen + critique rounds at 2× tokens), Online (one LoRA step
per episode on the model's own verified solution), Sleep (no updates by day;
every 50 episodes: verified dreams + 1:1 replay + 50 LoRA steps + 5% decay),
Sleep-NoDream. **Measures** every 50 episodes: 201 held-out puzzles, a 60-item
masked probe for the shortcut, 300 GSM8K items, secondary implicit metrics.
**Matching.** Generated tokens and gradient steps matched within ±5% across
regimes; `eval/analyze.py` refuses to compare unmatched seeds.

## Headline numbers (mean over 5 seeds)

| arm | held-out peak | held-out last-3 | probe (chance 0.33) | GSM8K Δ, grid regime | GSM8K, math regime |
|---|---|---|---|---|---|
| Baseline | 0.42 | 0.42 | 0.34 | 0 | 0.91 |
| Awake | 0.42 | 0.42 | 0.31 | 0 | — (no weights) |
| Online | 0.96 | 0.89 | 0.38 | +0.155 | 0.91 |
| Sleep | 0.94 | 0.89 | 0.42 | +0.031 | 0.93 |
| Sleep-NoDream | 0.95 | 0.92 | 0.38 | +0.079 | 0.92 |

Preregistered verdicts: H1–H3 (insight) not supported, all censored; H4
(retention) not supported, direction reversed (Sleep − Online GSM8K Δ −0.123,
p = 0.125, n = 5). See `records/analysis_final.md`. Why the last two columns
disagree is the subject of `records/followup/RESULTS.md`.

## Reproducing

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q                      # 142 model-free tests
# GSM8K test set (MIT, OpenAI) is not vendored: put the original test.jsonl at data/gsm8k_test.jsonl
python -m eval.analyze --final --results records --seeds 0,1,2,3,4   # preregistered analysis from the records
python -m eval.deep_analysis --results records --seeds 0,1,2,3,4      # exploratory analysis + figures
```

Running arms needs a GPU with ≥ 16 GB (bf16 Qwen3-4B + LoRA): `run_arm.py`
locally, `modal_app.py` on Modal, or `deploy/vultr/` on a plain VM. Every run
saves resumable state after each checkpoint. The 13 trained LoRA adapters
(Online, Sleep, Sleep-NoDream for seeds 1–4, plus Sleep-NoDream seed 0) are
published on Hugging Face: **rushiljain/doze-llm-adapters** (PEFT format;
load with `PeftModel.from_pretrained(base, path)`; `deploy/verify_adapter.py`
checks one). The seed 0 pilot Sleep and Online adapters were never saved.

## Layout

```
tasks/        number_reduction.py (task, splits, prompts, scoring); fold_variant2.py (transfer puzzle)
arms/         harness (checkpoints, ledger, resume), baseline, awake, online, sleep
sleep/        C3 filter, replay buffer, two-step dreamer, night consolidation
backends/     HF Transformers + PEFT backend (grid), MLX backend (calibration only), scripted backend (tests)
eval/         analyze (preregistered), deep_analysis, stats, compute_ledger, control_bench, shortcut_detector,
              implicit metrics, followup (post-hoc checks), transfer_local / transfer_analysis
deploy/       Vultr VM scripts, adapter export/verify; modal_app.py at the root
records/      all data and analyses (see records/README.md)
tests/        model-free test suite
```

## Cost and compute

Grid: 25 runs, ≈ 182 GPU-hours on 2 × NVIDIA A16 (Vultr, ≈ $90) plus seed 0 on
Modal L4 and two seed-1 arms on a Lightning L4. Follow-up: ≈ $7 on Modal L4
plus a laptop (Apple silicon, 18 GB) for the transfer run. Details and the
incident log: `records/TIMELINE.md`, `records/STATUS.md`.

## Citation

See `CITATION.cff`. Code is MIT; adapters derive from Qwen3-4B (Apache 2.0);
GSM8K is MIT (OpenAI).
