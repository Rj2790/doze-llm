# A preregistered sleep cycle for a 4B language model: what the null taught us about measuring insight and forgetting

*doze-llm, September 2026. Code, data and every number: https://github.com/Rj2790/doze-llm.*

## Abstract

We preregistered a test of whether a sleep-like training schedule helps a small language model (Qwen3-4B) discover a hidden shortcut in a digit puzzle sooner than continuous updating or inference-time reasoning, at matched compute. Five regimes, five seeds, 25 runs, 600 episodes each. All four preregistered hypotheses came out null. A hypothesis-blind review of the completed data, followed by pre-registered follow-up checks, showed that two of the study's instruments could not have detected what they were built for: the masked-item insight probe cannot be passed by any model at this scale, even one told the rule or trained directly on the probe, and the GSM8K "forgetting" control had run under a system prompt that forbids explanation, under which accuracy tracks answer length (r = 0.93) and which a normal prompt lifts from 0.59 to 0.91 for base and every trained adapter alike. What the data do establish is narrower and clearer: verified self-distillation on the model's own correct solutions teaches a fixed procedure quickly (0.42 → 0.86–0.99 held-out in 200–450 steps), generally (0.84–0.89 on strings without the practice structure), and equally under continuous or batched schedules; self-generated variants and self-critique add nothing; and the same training entrenches the procedure, so that on a puzzle whose rule alternates every step the trained models start worse than the untrained one. We report the design, the verdicts, the review, the follow-up, and the instrument lessons, with the intention that the last are the useful part.

## 1. Motivation and design

Humans who practise the Number Reduction Task and then sleep are about three times more likely to notice its hidden rule the next day (Wagner et al., 2004). Language models are either frozen or updated continuously; nobody had built the obvious analogue, a scheduled offline phase with replay and self-generated practice, and measured whether the phase structure itself matters. We did, with the design frozen before any run (`PREREG.md`, v1.0, public gist) and every later change logged (`DEVIATIONS.md`).

**Task.** Number Reduction, length 12 over {1,4,9}. Fold left to right: same digits give that digit; different digits give the third. Eleven steps; the last result is the answer. The model is told the rule, shown a six-digit worked example, and must write all eleven comparisons, a STEPS line and an ANSWER line (~97 tokens, greedy, thinking off). Every training and evaluation string is drawn from the subset in which the 6th and 11th intermediate results coincide, so the answer is available five steps early; nothing in the prompt says so. Splits are prefix-disjoint: 801 train, 201 held-out, 60 probe, answer-stratified, per seed.

**Regimes**, each run for the same 600-puzzle sequence per seed:

1. *Baseline.* Frozen weights, one attempt per puzzle.
2. *Awake.* Frozen weights, two to three self-critique rounds per puzzle, token budget matched to Sleep's cumulative generation.
3. *Online.* After each attempt, if the answer is correct and at least 9 of 11 steps are correct (the "C3" filter), one LoRA step on that attempt.
4. *Sleep.* No updates by day. Every 50 puzzles a night: (a) the model proposes new strings by editing 1–3 of the last five digits of buffered puzzles, solves them, and keeps those it solves correctly ("dreams", verified against the solver, duplicates and held-out prefixes rejected); (b) 50 LoRA steps on new keeps + accepted dreams + an equal draw of replayed keeps; (c) LoRA weights shrunk 5%.
5. *Sleep-NoDream.* As 4 without (a).

LoRA r=16, α=32 on all attention and MLP projections (33M parameters), Adam 1e-4, batch 1, bf16 base, deterministic CUDA kernels after the pilot.

**Measures** at 13 checkpoints (episode 0 and every 50): accuracy on the 201 held-out puzzles; accuracy on 60 probe items with digits 8–12 masked, answerable only via the coincidence (chance 1/3; criterion 0.70 at two consecutive checkpoints); exact-match on a fixed 300-item GSM8K sample; and post-pilot implicit metrics (answer-only accuracy on structured vs unstructured strings, mirror bias among late errors, error position).

**Hypotheses.** H1: Sleep reaches the probe criterion first (log-rank vs each other arm). H2: phase structure matters (Sleep vs Online). H3: dreaming matters (Sleep vs NoDream). H4: Sleep forgets less on GSM8K than Online (paired exact sign-flip test on Δ from episode 0).

**Matching.** Generated tokens matched between Baseline and Online, and between Awake and Sleep; gradient steps and training tokens matched between Online, Sleep and NoDream; all within ±3% in every seed. The analysis code refuses to compare unmatched seeds.

**Execution.** Seed 0 on Modal L4 (pilot code for three arms; see DEVIATIONS), seed 1 partly on a Lightning L4, seeds 1–4 on a Vultr 2 × A16 VM: 182 GPU-hours, ≈ $90. Every run saved resumable state per checkpoint; the incident log is in `records/TIMELINE.md`.

## 2. Preregistered results

Every seed passed the matching check. Verdicts (`records/analysis_final.md`):

| hypothesis | result | verdict |
|---|---|---|
| H1 insight, Sleep first to criterion | 0 of 25 runs ever met the criterion; all censored at 600 | not supported |
| H2 phase structure | all censored | not supported |
| H3 dreaming | all censored | not supported |
| H4 retention | Sleep − Online GSM8K Δ: −0.077, −0.257, −0.050, −0.257, +0.023; mean −0.123, p = 0.125 | not supported, direction reversed |

Per-arm means over seeds:

| arm | held-out peak | held-out last-3 | GSM8K Δ (grid regime) | seeds Δ > 0 | probe max |
|---|---|---|---|---|---|
| Baseline | 0.421 | 0.421 | 0 | — | 0.34 |
| Awake | 0.419 | 0.419 | 0 | — | 0.31 |
| Online | 0.963 | 0.889 | +0.155 | 5/5 | 0.38 |
| Sleep | 0.937 | 0.894 | +0.031 | 3/5 | 0.42 |
| Sleep-NoDream | 0.950 | 0.917 | +0.079 | 4/5 | 0.38 |

Nothing separates the three trained arms on the task (paired differences 0.005–0.023, p ≥ 0.5). The frozen arms are flat. The probe never left chance in any run at any checkpoint.

## 3. Deep exploratory analysis

`records/analysis_deep.md` and `records/analysis_deep_notes.md`, not preregistered.

- **Probe, pooled.** 3,900 item-evaluations per trained arm: 0.32–0.34, ±0.015. The conservative final-checkpoint pooling (300 independent items per arm) gives 0.33–0.34.
- **Execution learned from the left.** Day-attempt accuracy 0.46 → 0.89–0.92 across the twelve 50-episode blocks; correct leading steps per attempt 6.6 → ~10 of 11; the first error moved from mid-chain (mode 5–6) to late (7–8). The untrained model's errors were digit-tracking slips, not rule errors. Working length never shrank (97 → 97 tokens).
- **Schedule.** Sleep lags Online mid-run (block-6 day accuracy 0.70 vs 0.84) and catches up; batched curves are smoother (held-out max drawdown 0.11 vs 0.18).
- **Nights.** Dream yield rose 0.14 → 0.32 as the buffer grew; duplicates were ~40% of proposals throughout; no dream touched a held-out or probe prefix in ~3,660 proposals; by night 12 dreams were ~40% of new training items. Sleep vs NoDream: task −0.023, GSM8K −0.047, both n.s., for +80% generated tokens.
- **Awake.** Two critique rounds on 77% of attempts; accuracy by rounds 0.40 / 0.41 / 0.42; day accuracy 0.410 vs Baseline 0.405.
- **Buffers.** 336–454 distinct puzzles, 42–57% of the training set, balanced answers, zero leakage.
- **Consistency.** Recomputing held-out accuracy from the saved per-item rows reproduces every reported number.
- **Power.** H4 needs ~9 seeds at the observed effect; n = 5 cannot go below p = 0.0625.

## 4. The hypothesis-blind review

Before drawing conclusions we re-read the whole experiment as if we did not know why it was run, and had a fresh session with no access to the project do the same independently (`records/analysis_outsider.md`, Appendices A–B). The two readings agreed on the ranking of effects and the independent one raised three points, each verified against the run files:

1. **GSM8K accuracy tracked response length.** A per-item token proxy correlated r = 0.91 with GSM8K accuracy across 156 trained checkpoints. The grid's control benchmark had been generated under the *task* system prompt ("output only the requested lines … no explanation"), because the harness backend carries one system prompt per run; the control module's own math prompt was never used (DEVIATIONS, 2026-09-12). The base model's 0.58 was therefore a suppressed-reasoning score.
2. **The masked probe could not have detected the shortcut.** Answer-only accuracy on *fully visible* structured puzzles is 0.31–0.39 for every regime at every checkpoint: no model here can compute six steps silently, so a model that knew r11 = r6 perfectly would still score ~1/3.
3. **The first-night GSM8K dip was Sleep-only** (5/5 seeds down; NoDream 4/5 up) on ~12-item train sets differing by a few dreams, too little to carry a causal story; and nights present exactly 50 examples, so from ~night 4 the 1:1 interleave described the pool, not the batch.

We wrote decision rules for the checks that would settle these (`records/followup_prereg.md`) before running them.

## 5. Follow-up experiments

`records/followup/RESULTS.md`; base model plus the 13 saved adapters; Modal L4; ≈ $7.

**A. GSM8K under two regimes, outputs saved.** The grid's regime reproduces to ±0.02 on every model. Under a math system prompt with 1024 tokens: base **0.910**; every adapter 0.897–0.933 (Online −0.001, NoDream +0.011, Sleep +0.015 vs base, means). Accuracy–length correlation across models: 0.93 in the grid's regime, 0.22 in the math regime. Nothing was forgotten and nothing was gained; the H4 axis measured verbosity.

**B. Probe instrument.** Answer-only probe with the rule stated in the prompt: chance for every model (0.29–0.35). Fresh LoRA trained 200 steps on masked-prompt → answer pairs: probe 0.32–0.35, training loss floor ≈ 0.21, with a small cost to chain accuracy. Teacher-forced log-probability argmax: chance for all 14 models, negative margins. With working allowed and the rule stated, Online reaches 0.78 — but the last written visible result *is* r6, so that variant measures computation, not shortcut knowledge. H1–H3 were unobservable as designed and unreachable at this scale.

**C. Generalisation.** Unstructured strings (no mirror structure): base 0.372, Online 0.844, Sleep 0.840, NoDream 0.887, against 0.436 / 0.913 / 0.888 / 0.925 on structured held-out; the base model's own gap is 0.06. General execution learning, not specialisation to the practice structure.

**D. Transfer to a rule that alternates** (`records/transfer/RESULTS.md`; four symbols, odd steps take the smaller uninvolved symbol, even steps the larger; 201 items × 14 models; laptop). All models at chance (0.20–0.34; chance 0.25). Base gets step 1 right 90% of the time and step 2 34%; the adapters are worse at step 1 (Online 0.61, Sleep 0.79, NoDream 0.84). Old-table leakage is low (6–19% of first rule errors); the dominant errors are applying the other step's table or copying an operand. The training entrenched "one rule, every step".

## 6. What we conclude

1. Verified self-distillation on a model's own correct chains repairs an execution-limited skill within a few hundred single-example LoRA steps, generalises within the rule, and does so equally whether the steps are taken one per episode or fifty per night. The schedule changes the path (batched lags, then is smoother), not the destination.
2. Self-generated variants with an exact verifier added nothing at up to 40% of night data; self-critique without external feedback added nothing at 2× tokens.
3. The same training entrenches the procedure and transfers negatively to a rule that changes with position.
4. The study cannot speak to insight. Its probe demanded silent computation the model cannot do, and no regime rewarded the shortcut. Its retention result was an artefact of the control prompt. Both are documented with evidence rather than asserted.

## 7. Instrument lessons

- **Make the probe passable and prove it.** A positive control (a model that knows the rule must pass) should be part of the preregistration, and the probe should allow the model to write what it can compute, with a no-structure control set to separate "uses the shortcut" from "stops where the digits stop".
- **Give the shortcut a payoff.** Supervised training on full correct chains reinforces chain execution. If discovery is the question, the regime must reward shorter or earlier answers.
- **Run control benchmarks under their own prompt and save the outputs.** One system prompt per backend is a footgun; a 0.58 GSM8K baseline for a model that scores 0.91 should have been caught at calibration.
- **Pool wisely.** Frozen arms repeat one evaluation at every checkpoint; 300-item benchmarks have ±0.03 standard errors; five seeds give a sign-flip test a floor of p = 0.0625.
- **Keep the artefacts.** Per-item rows, full trajectories, ledgers and resumable state made every number in this report recomputable and made the review possible.

## 8. Reproducibility

Everything is in the repository: frozen preregistration, deviations, project log, all 25 run files with per-episode texts and per-checkpoint per-item rows, compute ledgers, logs, timing, final replay buffers, the analyses and figures, and the follow-up outputs including every GSM8K response. The 13 trained adapters are on Hugging Face (`rushiljain/doze-llm-adapters`). The analysis pipelines are re-runnable from `records/` alone. Model-free tests cover the task, filters, buffer, dreamer, ledger, matching gate, statistics and analysis code.

## References

Wagner, U., Gais, S., Haider, H., Verleger, R., & Born, J. (2004). Sleep inspires insight. *Nature*, 427, 352–355.
Qwen Team (2025). Qwen3 technical report. Cobbe, K. et al. (2021). Training verifiers to solve math word problems (GSM8K).
